# `agent_forge/src/agent_forge/consciousness`

<!-- EIDOS:DOCSYS:START -->
- Contract: `eidos.directory_doc.v1`
- Generated: `2026-03-13T00:19:21Z`
- Path: `agent_forge/src/agent_forge/consciousness`

## What It Is

Consciousness runtime, metrics, perturbation, and benchmark substrate used by Eidos runtime status and experiments.

## Why It Exists

This package supplies the operational continuity, bridge state, and evaluative signals that the rest of the control plane depends on.

## How It Works

- Tracked files in scope: `59`
- Child directories: `5`
- Tests detected: `False`
- Python modules: `benchmarks, external_adapters, features, index, integrated_benchmark, kernel, ledger, linking, protocol, state_store`

## Contents

- [`bench`](./bench/README.md)
- [`metrics`](./metrics/README.md)
- [`modules`](./modules/README.md)
- [`perturb`](./perturb/README.md)
- [`tuning`](./tuning/README.md)

## Prominent Files

- [`__init__.py`](./__init__.py)
- [`benchmarks.py`](./benchmarks.py)
- [`external_adapters.py`](./external_adapters.py)
- [`features.py`](./features.py)
- [`index.py`](./index.py)
- [`integrated_benchmark.py`](./integrated_benchmark.py)
- [`kernel.py`](./kernel.py)
- [`ledger.py`](./ledger.py)

## Strengths

- The directory exposes importable Python modules rather than only opaque assets.
- Responsibility is split into child directories instead of one flat file heap.

## Weaknesses / Risks

- No directly associated test coverage was detected under the tracked file set for this directory.

## Next Steps

- Keep bridge and memory defaults aligned with the canonical vector-native stores.
- Expand explicit evidence lineage and contradiction analysis in the graph-facing outputs.

## References

- Parent README: [`agent_forge/src/agent_forge/README.md`](../README.md)
- Forge README: [`agent_forge/README.md`](../../../README.md)

## Accuracy Contract

- This README is generated from tracked repository structure and conservative code scanning.
- Behavior claims should be backed by tests, routes, or directly linked source files.
<!-- EIDOS:DOCSYS:END -->
