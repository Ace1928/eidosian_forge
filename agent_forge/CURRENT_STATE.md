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
  - interventional validity summary (perturbation effect consistency),
  - security boundary summary,
  - protocol compatibility checks,
  - explicit gate outcomes.
- Protocol surfaces:
  - machine-readable schema artifact (`docs/consciousness/rac_ap_protocol_v1_2026_02_19.json`),
  - CLI protocol validation/template export,
  - CLI pre-registration manifest generation.

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
- `eidctl consciousness protocol`
- `eidctl consciousness preregister`

## Current Gaps
- RAC-AP validation currently depends on sufficient historical report volume.
- Security gate is available but currently optional by protocol default.
- External benchmark adapters and MCP validation resource integration are pending.
