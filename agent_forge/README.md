# Agent Forge (Eidos E3)

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Minimal scaffolding for the Eidos E3 system.

## CLI tools
- `bin/eidctl` – inspect state snapshots and manage the JSONL journal.
- `bin/eidctl workspace` – summarize workspace broadcasts and ignition windows.
- `bin/eidctl self-model` – snapshot memory + workspace signals (optional emit).
- `bin/eidosd` – daemon shim; `--once` runs one beat, `--loop` runs a scheduler collecting metrics
- `bin/eidtop` – curses TUI for live beats and metrics.

## Core modules
- `core/config.py` – strict YAML config loader.
- `core/state.py` – state migrations, journal, snapshots.
- `core/events.py` – append-only event bus with rotation.
- `core/db.py` – SQLite helpers for metrics and journal tables.

## Bootstrap
Run `scripts/bootstrap.sh` to seed configs, create a virtual environment and
install dependencies.

## Testing

Run the test suite with the repository's virtual environment to ensure
dependencies like PyYAML are present:

```
.venv/bin/python -m pytest -q
```
## Smoke test

Run `make smoke` for an end-to-end verification of the bootstrap, state,
journal, and test flow.

## Daemon loop

Run the scheduler:

```
bin/eidosd --state-dir state --loop --tick 5
```

## eidtop

```
bin/eidtop --state-dir state
```
Keys: `q` quit, `r` refresh, `p` pause, `m` cycle metric, `?` help.

## Retention

Add to `cfg/self.yaml` to control automatic pruning:

```
daemon:
  retention:
    metrics_per_key_max: 10000
    events_keep_days: 7
    journal_max_bytes: 5242880
    maintenance_every_beats: 100
```

The daemon runs maintenance every `maintenance_every_beats` beats, pruning
SQLite metrics, rotating the journal, and removing old event directories.
