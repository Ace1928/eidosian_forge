# Current State: archive_forge

**Date**: 2026-01-20
**Status**: Archived / Static

## 📊 Metrics
- **Files**: ~100+ Python scripts and experiments.
- **Maintenance**: None.

## 🏗️ Architecture
Unstructured flat list of files + some directories (`code/`, `notebooks/`).

## 🐛 Known Issues
- Everything here is potentially broken.

## 🔄 Recent Updates
- Added an archive pipeline that extracts and normalizes symbols into
  `archive/normalized/` with metadata in `archive/metadata/`.
- GraphRAG export files are generated under `archive/metadata/graphrag/`.
