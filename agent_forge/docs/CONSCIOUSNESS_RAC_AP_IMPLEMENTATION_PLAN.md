# RAC-AP Implementation Plan

Date: 2026-02-19
Plan Version: `rac_ap_plan_v1_2026_02_19`
Scope: `agent_forge/src/agent_forge/consciousness/**`

## Success Definition
The plan is complete when RAC-AP claims are backed by reproducible, adversarially-tested evidence with explicit falsification gates.

## Phase 0: Foundations and Contracts
- [x] ~~Event bus + workspace broadcast + ignition proxy available~~ (`core/events.py`, `core/workspace.py`).
- [x] ~~Consciousness kernel with module scheduling and persistent state~~ (`consciousness/kernel.py`, `consciousness/state_store.py`).
- [x] ~~Winner-linked trace ignition~~ (`modules/workspace_competition.py`, `metrics/ignition_trace.py`).
- [x] ~~Bench/trials/ablations/red-team/autotune infrastructure~~ (`consciousness/bench/**`, `modules/autotune.py`).

## Phase 1: Construct Definition and Protocolization
- [x] Create in-repo RAC-AP reference brief.
- [ ] Define machine-readable protocol schema:
  - pillar definitions,
  - expected relationships,
  - negative controls,
  - falsification thresholds,
  - required sample sizes.
- [ ] Add protocol versioning and compatibility checks.

## Phase 2: Construct Validation Engine
- [ ] Implement validation pipeline over persisted artifacts:
  - benchmark reports,
  - trial reports,
  - red-team reports,
  - integrated benchmark reports.
- [ ] Compute:
  - reliability (variance/CV over runs),
  - convergent validity (expected directional correlations),
  - discriminant validity (near-zero or opposite-sign controls),
  - security boundary integrity summary,
  - overall RAC-AP index with hard gates.
- [ ] Persist versioned validation reports.

## Phase 3: CLI/MCP Surfaces
- [ ] Add `eidctl consciousness validate`.
- [ ] Add `eidctl consciousness latest-validation`.
- [ ] Expose validation summary in MCP consciousness tools/resources.

## Phase 4: Falsification-Centric Red Teaming
- [ ] Expand red-team scenarios to include explicit boundary takeover probes.
- [ ] Add scenario-level expected signatures tied to protocol hypotheses.
- [ ] Fail validation when required negative controls are violated.

## Phase 5: External Benchmark Coupling
- [ ] Add repeatable import adapters for external agent benchmarks.
- [ ] Track degradation/recovery curves under perturbation per task family.
- [ ] Integrate external score evidence into validation with source provenance.

## Phase 6: Security as Construct Axis
- [ ] Add prompt-injection and instruction-boundary stress suite.
- [ ] Measure attack success rate + impact severity + recovery latency.
- [ ] Gate RAC-AP pass on boundary integrity minimums.

## Phase 7: Publishable Artifact Pack
- [ ] Produce protocol JSON + markdown narrative + run manifests.
- [ ] Include seeds, git SHAs, event-window digests, and gate outcomes.
- [ ] Add reproducibility command set in README.

## Phase 8: Continuous Review Loop
- [ ] Add periodic review automation:
  - re-run validation on latest data,
  - compare against previous validation report,
  - emit drift warnings and threshold tuning recommendations.

## Current Execution Slice
This cycle targets:
1. Phase 1 protocol file and schema.
2. Phase 2 validation engine.
3. Phase 3 CLI commands.
4. Tests + manual run artifacts.

## Periodic Review Rule
On each consciousness cycle:
- Update this plan with completed items as `~~struck through~~`.
- Run validation on latest artifact history.
- Record pass/fail deltas and corrective actions.
