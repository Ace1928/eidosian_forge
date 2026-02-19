# Current State: agent_forge

Date: 2026-02-19
Status: Active development, consciousness runtime production operational.

## Implemented
- Core agent orchestration and event-sourced runtime (`core/*`).
- Consciousness kernel with modular multi-rate execution (`consciousness/kernel.py`).
- GNW-style competition and winner-linked ignition tracing.
- Predictive world model, intero/affect modulation, self-model extensions.
- Bench stack: trial runner, ablations, red-team, stress benchmark, integrated benchmark.
- Autotune and experiment-designer modules.
- RAC-AP construct validation engine with:
  - reliability summary,
  - nomological expectation checks,
  - convergent/discriminant scores,
  - security boundary summary,
  - explicit gate outcomes.

## Primary Docs
- `agent_forge/docs/CONSCIOUSNESS_RAC_AP_RESEARCH_2026-02-19.md`
- `agent_forge/docs/CONSCIOUSNESS_RAC_AP_IMPLEMENTATION_PLAN.md`

## Primary CLI Commands
- `eidctl consciousness status`
- `eidctl consciousness trial`
- `eidctl consciousness benchmark`
- `eidctl consciousness red-team`
- `eidctl consciousness full-benchmark`
- `eidctl consciousness validate`
- `eidctl consciousness latest-validation`

## Current Gaps
- RAC-AP validation currently depends on sufficient historical report volume.
- Security gate is available but currently optional by protocol default.
- External benchmark adapters and protocol pre-registration automation are pending.
