# 🤖 Agent Forge (Eidos E3)

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Autonomous Executors of the Eidosian System.**

## 🧠 Overview

`agent_forge` provides the scaffolding and runtime environment for autonomous agents within Eidos. While `eidos_mcp` provides the tools and context (the "Brain"), `agent_forge` provides the agency and execution loops (the "Hands").

It implements the **Eidos E3** architecture: **E**vent-driven, **E**pisodic, **E**volutionary.

## 🏗️ Architecture

- **Core Loop (`bin/eidosd`)**: The heartbeat of the agent. It cycles through perception, decision, and action phases.
- **State Management (`core/state.py`)**: Handles the journal (JSONL) and state snapshots, ensuring persistence and rollback capabilities.
- **Event Bus (`core/events.py`)**: An append-only log of all agent perceptions and actions.

## 🔧 CLI Tools

- `bin/eidctl`: Inspect state snapshots and manage the JSONL journal.
- `bin/eidctl workspace`: Summarize workspace broadcasts and ignition windows.
- `bin/eidctl self-model`: Snapshot memory + workspace signals.
- `bin/eidosd`: Daemon shim; `--once` runs one beat, `--loop` runs the scheduler.
- `bin/eidtop`: Curses TUI for monitoring live beats and metrics.

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

Run the test suite:
```bash
../eidosian_venv/bin/python -m pytest -q
```

## 🧹 Retention & Maintenance

Configure automatic pruning in `cfg/self.yaml`:

```yaml
daemon:
  retention:
    metrics_per_key_max: 10000
    events_keep_days: 7
    journal_max_bytes: 5242880
    maintenance_every_beats: 100
```
