# Agent Forge

Agent Forge is the runtime control layer for Eidosian agents.
It now includes a full consciousness instrumentation stack with perturbation testing, ablations, red-team campaigns, and integrated benchmark reporting.

## Core Surfaces
- `src/agent_forge/core/`: event bus, workspace, state snapshots, daemon utilities.
- `src/agent_forge/consciousness/`: kernel, modules, metrics, perturbations, tuning, and benchmark suites.
- `src/agent_forge/cli/eidctl.py`: operational CLI for status, trials, benchmarks, and runtime diagnostics.
- `bin/eidctl`, `bin/eidosd`, `bin/eidtop`: shell-facing control tools.

## Consciousness Stack
Key implemented capabilities:
- GNW-style attention -> competition -> winner broadcast loops.
- Winner-linked ignition tracing with explicit trace-strength gates.
- Predictive world-model metrics and self/boundary monitoring.
- Phenomenology proxy metrics (unity, continuity, ownership, perspective coherence, dream-likeness).
- Perturbation suite and trial harness with replayable artifacts.
- Ablation matrix, red-team campaign, stress benchmark, integrated benchmark.

## RAC-AP Research and Plan
- Research brief: `agent_forge/docs/CONSCIOUSNESS_RAC_AP_RESEARCH_2026-02-19.md`
- Implementation plan: `agent_forge/docs/CONSCIOUSNESS_RAC_AP_IMPLEMENTATION_PLAN.md`

## CLI Examples
```bash
# Runtime status
python agent_forge/bin/eidctl consciousness status --dir state --json

# Perturbation trial
python agent_forge/bin/eidctl consciousness trial --dir state --kind noise --target attention --ticks 3 --json

# Benchmark suite
python agent_forge/bin/eidctl consciousness benchmark --dir state --ticks 12 --json

# RAC-AP construct validation
python agent_forge/bin/eidctl consciousness validate --dir state --limit 64 --min-pairs 6 --json

# Red-team campaign
python agent_forge/bin/eidctl consciousness red-team --dir state --quick --json

# Integrated benchmark
python agent_forge/bin/eidctl consciousness full-benchmark --dir state --skip-llm --skip-mcp --json
```

## Testing
```bash
./eidosian_venv/bin/python -m pytest agent_forge/tests -q
```

## Engineering Rules
- Prefer append-only events and explicit provenance.
- Treat every metric claim as a falsifiable hypothesis.
- Keep perturbation and ablation checks in CI-facing tests.
- Do not equate fluent output with validated coherence.
