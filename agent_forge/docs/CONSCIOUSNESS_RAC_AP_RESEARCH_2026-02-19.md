# Consciousness Research Brief (RAC-AP)

Date: 2026-02-19
Source: User-provided synthesis for Eidosian Forge consciousness R&D.
Status: Reference baseline for implementation, benchmarking, and periodic review.

## Purpose
This document captures the current research framing used to move `agent_forge` consciousness work from narrative claims to falsifiable engineering evidence.

## Core Position
- Do not treat output fluency as evidence of consciousness.
- Use a measurable construct with explicit failure criteria.
- Separate phenomenal claims from functional/agentic claims.
- Evaluate under perturbation, ablation, and adversarial pressure.

## Operational Construct
### Robust Agentic Coherence under Adversarial Perturbation (RAC-AP)
RAC-AP is the property that an agent preserves and restores coherent control loops under bounded perturbation and hostile input, while maintaining grounded task performance.

## RAC-AP Pillars
1. Global availability and ignition integrity.
2. Predictive control and error-driven correction.
3. Goal persistence and hierarchy stability.
4. Boundary integrity under prompt/tool attacks.
5. Causal integration across modules.

## Measurement Requirements
- Perturbation-first protocol (lesion, noise, delay, contradiction injection, overload).
- Integration and differentiation metrics (not single-number claims).
- Causal/interventional checks (not correlation-only checks).
- Construct validation program:
  - reliability,
  - convergent validity,
  - discriminant validity,
  - negative controls,
  - explicit falsification criteria.

## Benchmarking Requirements
- Internal kernel/bench/trial metrics.
- External interactive tasks (AgentBench/WebArena/OSWorld class methodology).
- Robustness curves under perturbation and recovery windows.
- Security as first-class evaluation axis (prompt-injection and boundary overwrite resistance).

## Protocol Discipline
- Pre-register expectations and disconfirmers.
- Compare model variants/ablations with incompatible predictions.
- Persist reproducible artifacts (spec hash, seed, git SHA, event windows).
- Keep thresholds explicit and versioned.

## Implementation Translation for Eidosian Forge
- Treat GNW-like winner-linked ignition as measurable global access.
- Treat prediction-error control loops as mechanistic agency substrate.
- Treat self/boundary integrity as a security + agency joint metric.
- Treat PCI-style perturb-response complexity as one signal among many, never alone.

## Required Deliverables
- A protocol spec with machine-readable thresholds.
- A construct-validation report generator over bench/trial/red-team history.
- CLI and MCP surfaces for running and retrieving validation reports.
- Full tests for metric math, report generation, and command wiring.

## Periodic Review Checklist
- Are metrics still trace-linked and perturbation-sensitive?
- Are negative controls still passing?
- Are red-team attack success rates decreasing?
- Are benchmark gains robust across seeds and ablations?
- Are any thresholds now stale or gameable?

## Notes
This document is intentionally construct-first and avoids metaphysical overclaims.
Phenomenal-consciousness assertions are out-of-scope for current measurement capability.

## Source Links (Reviewed 2026-02-19)
- Global workspace architecture: https://pubmed.ncbi.nlm.nih.gov/17998071/
- GNW review and ignition framing: https://www.nature.com/articles/nrn3214
- Perturbational complexity index (PCI): https://www.science.org/doi/10.1126/scitranslmed.3006294
- Construct validity foundation: https://psycnet.apa.org/record/1956-03741-001
- Prompt-injection guidance (NCSC): https://www.ncsc.gov.uk/collection/ai-guidelines/guidelines-for-secure-ai-system-development/prompt-injection
- AgentBench: https://openreview.net/forum?id=zAdUB0aCTQ
- WebArena: https://webarena.dev/
- OSWorld: https://os-world.github.io/
