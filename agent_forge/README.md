# 🤖 Agent Forge (Eidos E3)

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Autonomous Executors of the Eidosian System.**

## 🧠 Overview

`agent_forge` provides the scaffolding and runtime environment for autonomous agents within Eidos. While `eidos_mcp` provides the tools and context (the "Brain"), `agent_forge` provides the agency, execution loops, and consciousness kernel (the "Hands" and "Self").

It implements the **Eidos E3** architecture:
- **E**vent-driven: All actions and perceptions are immutable events.
- **E**pisodic: State is checkpointed and recoverable.
- **E**volutionary: Agents adapt via the Consciousness Kernel.

## 🏗️ Architecture

### Core Modules
- `core/config.py`: Strict YAML config loader (schema validation).
- `core/state.py`: State migrations, journal (JSONL), and snapshot management.
- `core/events.py`: Append-only event bus with rotation policies.
- `core/db.py`: SQLite helpers for time-series metrics and journal tables.
- `core/scheduler.py`: Robust loop handling with jitter and backoff.
- `consciousness/`: The `ConsciousnessKernel` responsible for self-reflection and perturbation trials.

### The Loop (`bin/eidosd`)
The daemon runs a heartbeat loop (`--tick N`) that:
1.  Collects OS and process metrics.
2.  Ticks the **Consciousness Kernel**.
3.  Processes the Goal/Step scheduler.
4.  Performs periodic maintenance (pruning, rotation).

## 🔧 CLI Tools

### `eidctl`
The primary control interface.
- `eidctl state`: Print current state snapshot.
- `eidctl journal`: Append or inspect the append-only journal.
- `eidctl goals`: Manage agent goals (CRUD).
- `eidctl steps`: Inspect execution steps for goals.
- `eidctl runs`: Inspect specific execution runs.
- `eidctl workspace`: Summarize workspace broadcasts and ignition windows.
- `eidctl self-model`: Snapshot memory + workspace signals.
- `eidctl consciousness`: Inspect kernel status and perturbation trials.

### `eidosd`
The daemon shim.
- `--once`: Run one beat and exit (useful for cron/testing).
- `--loop`: Run the scheduler loop.
- `--tick SECS`: Seconds between beats (default: 5).
- `--jitter-ms MS`: Random jitter to avoid thundering herds.

### `eidtop`
Curses TUI for live monitoring.
- **Keys**:
    - `q`: Quit.
    - `r`: Refresh.
    - `p`: Pause/Resume updates.
    - `m`: Cycle metric view.
    - `?`: Toggle help.

## 🔗 System Integration

Agents running in `agent_forge` consume tools provided by **`eidos_mcp`**.
They are documented recursively by **`doc_forge`**.

## 🚀 Quickstart

1. **Bootstrap**:
   ```bash
   scripts/bootstrap.sh
   ```
   Seeds configs, creates venv, and installs dependencies.

2. **Run Daemon**:
   ```bash
   bin/eidosd --state-dir state --loop --tick 5
   ```

3. **Monitor**:
   ```bash
   bin/eidtop --state-dir state
   ```

## 🧪 Testing

Run the test suite via the virtual environment:
```bash
../eidosian_venv/bin/python -m pytest -q
```

**Smoke Test**:
Run `make smoke` (if Make is available) for an end-to-end verification of bootstrap, state, journal, and test flow.

## 🧹 Retention & Maintenance

Configure automatic pruning in `cfg/self.yaml`. The daemon runs maintenance every `maintenance_every_beats` cycles.

```yaml
daemon:
  retention:
    metrics_per_key_max: 10000
    events_keep_days: 7
    journal_max_bytes: 5242880
    maintenance_every_beats: 100
```
