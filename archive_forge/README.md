# Archive Forge

**The Museum of Eidos.**

## 🏛️ Overview

`archive_forge` preserves legacy code, experiments, one-off scripts, and early
prototypes while providing a pipeline to extract canonical entities into a
clean, queryable archive.

Legacy artifacts are **not guaranteed to run** and should be treated as
historical reference material. The pipeline extracts classes, functions, and
symbols into standardized, documented outputs.

## 🧭 Archive Layout

- `staging/`: Legacy intake queue (manual artifacts, pre-processed files).
- `canonical/`: Legacy canonical fragments produced by the original deconstructor.
- `archive/raw/`: Raw snapshots of scanned files (copy-on-write).
- `archive/normalized/`: Normalized entities (one file per symbol).
- `archive/metadata/`: Manifests, indices, and GraphRAG export files.
- `code/`, `images/`, `papers/`: Archived source material and assets.

## ✅ Pipeline Usage

Run the pipeline from the repo root:

```bash
python -m archive_forge --no-llm
```

Optional flags:

```bash
python -m archive_forge --scan-root ./archive_forge/staging --no-llm
python -m archive_forge --sync-knowledge --sync-graphrag
```

## ⚠️ Warnings

- Do not import legacy content in production code.
- Dependencies in archived scripts may be outdated or conflicting.
- The pipeline outputs are safe to consume, but legacy sources may be unstable.

## 📂 Notable Artifacts

- `qwen*.py`: Experiments with Qwen models.
- `particles*.py`: Early simulation tests.
- `notebooks/`: Research notebooks.
