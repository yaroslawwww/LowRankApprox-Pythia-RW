import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import RefinedWebCacheConfig, cache_refinedweb_subset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache a streaming RefinedWeb subset into JSONL shards.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="tiiuae/falcon-refinedweb")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--max-gb", type=float, default=None)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--shard-gb", type=float, default=0.5)
    parser.add_argument("--shuffle-buffer-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    max_bytes = None
    if args.max_gb is not None:
        max_bytes = int(args.max_gb * 1024**3)

    config = RefinedWebCacheConfig(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        split=args.split,
        text_column=args.text_column,
        max_bytes=max_bytes,
        max_documents=args.max_documents,
        shard_size_bytes=int(args.shard_gb * 1024**3),
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=args.seed,
        token=args.token,
    )
    stats = cache_refinedweb_subset(config)
    print(json.dumps(stats.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
