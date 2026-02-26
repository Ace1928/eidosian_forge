# Eidosian Forge

Brutal clarity, production-first engineering, and modular agent infrastructure in one monorepo.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](README.md)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/Ace1928/eidosian_forge/workflows/Eidosian%20Universal%20CI/badge.svg)](https://github.com/Ace1928/eidosian_forge/actions/workflows/ci.yml)

## What This Repo Is

`eidosian_forge` is a unified Forge stack:

- **Eidosian Atlas** (`web_interface_forge/EIDOS_ATLAS.md`): Real-time dashboard for system telemetry, documentation status, and knowledge base visualization.
- Runtime orchestration (`agent_forge`, `eidos_mcp`, `memory_forge`, `knowledge_forge`)
- Capability forges (code, docs, diagnostics, terminal, metadata, testing, visualization, and more)
- Shared libraries, scripts, workflows, reports, and state artifacts

Each `*_forge` directory is a composable module with its own source tree and test surface.

## Start Here

1. Clone and enter the repository.
2. Use the canonical environment (`eidosian_venv`) for all project commands.
3. Run status and tests before making changes.

```bash
git clone https://github.com/Ace1928/eidosian_forge.git
cd eidosian_forge
source eidosian_venv/bin/activate

python bin/eidosian status
PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src python -m pytest -q
```

## Documentation Hub

| Topic | Path |
| --- | --- |
| Docs portal | `docs/README.md` |
| Quickstart | `QUICKSTART.md` |
| Architecture | `ARCHITECTURE.md` |
| Forge architecture notes | `docs/FORGE_ARCHITECTURE.md` |
| Workflows quickstart | `WORKFLOWS_QUICKSTART.md` |
| Consciousness implementation docs | `docs/consciousness_fcl/README.md` |
| Port registry | `docs/PORT_REGISTRY.md` |
| Canonical model selection | `docs/MODEL_SELECTION.md` |
| Living knowledge system | `docs/LIVING_KNOWLEDGE_SYSTEM.md` |
| Security docs | `docs/security/` |
| CI/workflow docs | `.github/workflows/README.md` |

## Directory Coverage

Repository directory documentation is now explicit and complete:

- User-friendly atlas (top-level + second-level, linked): `docs/DIRECTORY_ATLAS.md`
- Full recursive directory index (all detected directories): `docs/DIRECTORY_INDEX_FULL.txt`
- Generator script: `scripts/generate_directory_atlas.py`

Regenerate directory docs at any time:

```bash
./eidosian_venv/bin/python scripts/generate_directory_atlas.py \
  --repo-root . \
  --atlas-output docs/DIRECTORY_ATLAS.md \
  --full-output docs/DIRECTORY_INDEX_FULL.txt \
  --max-depth 2 \
  --scope tracked

# Optional local runtime view (includes non-tracked local dirs)
./eidosian_venv/bin/python scripts/generate_directory_atlas.py \
  --repo-root . \
  --atlas-output docs/DIRECTORY_ATLAS.md \
  --full-output docs/DIRECTORY_INDEX_FULL.txt \
  --max-depth 2 \
  --scope filesystem \
  --include-hidden-top-level
```

## Core Runtime Commands

### Unified CLI Hub

```bash
python bin/eidosian status
python bin/eidosian forges
python bin/eidosian memory status
python bin/eidosian knowledge search "architecture"
python bin/eidosian code analyze README.md
```

### Consciousness Runtime and Benchmarks

```bash
PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src \
./eidosian_venv/bin/python -m agent_forge.cli.eidctl consciousness status --json

PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src \
./eidosian_venv/bin/python -m agent_forge.cli.eidctl consciousness benchmark --ticks 40 --json

PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src \
./eidosian_venv/bin/python -m agent_forge.cli.eidctl consciousness stress-benchmark --ticks 40 --json

PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src \
./eidosian_venv/bin/python -m agent_forge.cli.eidctl consciousness full-benchmark \
  --rounds 2 --bench-ticks 30 --trial-ticks 20 --skip-llm --red-team-quick --json
```

## Testing and Quality Gates

```bash
# Core consciousness + ops regression suite
PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src \
./eidosian_venv/bin/python -m pytest -q \
  agent_forge/tests/test_consciousness_*.py \
  agent_forge/tests/test_events_corr.py \
  scripts/tests/test_consciousness_benchmark_trend.py \
  scripts/tests/test_linux_audit_matrix.py

# MCP tool call regression
PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src \
./eidosian_venv/bin/python -m pytest -q eidos_mcp/tests/test_mcp_tool_calls_individual.py

# Forge-wide smoke audit (per-forge pytest --maxfail=1 with timeout controls)
./scripts/run_forge_test_audit.sh
```

## Major Top-Level Areas

| Group | Directories |
| --- | --- |
| Core runtime | `agent_forge`, `eidos_mcp`, `memory_forge`, `knowledge_forge`, `ollama_forge` |
| Engineering forges | `code_forge`, `doc_forge`, `diagnostics_forge`, `test_forge`, `terminal_forge`, `metadata_forge`, `refactor_forge`, `repo_forge` |
| Product/domain forges | `file_forge`, `word_forge`, `glyph_forge`, `llm_forge`, `prompt_forge`, `viz_forge`, `web_interface_forge`, `game_forge`, `lyrics_forge`, `sms_forge`, `mkey_forge`, `narrative_forge`, `erais_forge`, `type_forge`, `version_forge`, `archive_forge`, `article_forge`, `computer_control_forge`, `crawl_forge`, `audit_forge`, `figlet_forge`, `gis_forge`, `moltbook_forge` |
| Shared/core infra | `lib`, `bin`, `scripts`, `requirements`, `.github`, `docs` |
| Runtime data | `data`, `state`, `reports`, `logs`, `audit_data`, `Backups` |

For exhaustive per-directory detail, use `docs/DIRECTORY_ATLAS.md` and `docs/DIRECTORY_INDEX_FULL.txt`.
