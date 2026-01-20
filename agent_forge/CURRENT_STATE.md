# Current State: agent_forge

**Date**: 2026-01-20
**Identity**: "Eidos E3" (Evolution 3?)
**Status**: Active / Experimental

## 📊 Metrics
- **Build System**: Uses `uv` (modern Python tool) and `hatchling`.
- **Key Components**:
    - `bin/`: CLI tools (`eidctl`, `eidosd`, `eidtop`).
    - `core/`: State management (SQLite), Events, Config.
    - `actuators/`: Tools for agents?
    - `planners/`: Planning logic?
- **Testing**: `pytest` config exists.

## 🏗️ Architecture
Appears to be a daemon-based agent runner (`eidosd`) with a TUI (`eidtop`) for monitoring.
It uses a "Journal" concept (`core/state.py`) for persistence.

## 🐛 Known Issues
- Split between `core/` and `src/` needs investigation.
- Dependencies in `pyproject.toml` were minimal.