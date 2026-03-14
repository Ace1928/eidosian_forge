# `agent_forge/src/agent_forge/consciousness/modules`

<!-- EIDOS:DOCSYS:START -->
- Contract: `eidos.directory_doc.v1`
- Generated: `2026-03-13T00:49:19Z`
- Path: `agent_forge/src/agent_forge/consciousness/modules`

## What It Is

Managed directory documentation for `agent_forge/src/agent_forge/consciousness/modules`, generated from tracked files and directory structure.

## Why It Exists

This directory exists to hold the `forge` surface for `agent_forge` and keep that responsibility separate from adjacent forge concerns.

## How It Works

- Tracked files in scope: `19`
- Child directories: `0`
- Tests detected: `True`
- Python modules: `affect, attention, autotune, experiment_designer, intero, knowledge_bridge, memory_bridge, meta, phenomenology_probe, policy`

## Contents

- No managed child directories detected.

## Prominent Files

- [`README.md`](./README.md)
- [`__init__.py`](./__init__.py)
- [`affect.py`](./affect.py)
- [`attention.py`](./attention.py)
- [`autotune.py`](./autotune.py)
- [`experiment_designer.py`](./experiment_designer.py)
- [`intero.py`](./intero.py)
- [`knowledge_bridge.py`](./knowledge_bridge.py)

## Validating Tests

- [`agent_forge/tests/test_consciousness_memory_knowledge_bridge.py`](../../../../tests/test_consciousness_memory_knowledge_bridge.py)
- [`agent_forge/tests/test_workspace.py`](../../../../tests/test_workspace.py)
- [`agent_forge/tests/test_agent.py`](../../../../tests/test_agent.py)
- [`agent_forge/tests/test_consciousness_attention_competition_learning.py`](../../../../tests/test_consciousness_attention_competition_learning.py)
- [`agent_forge/tests/test_consciousness_experiment_designer.py`](../../../../tests/test_consciousness_experiment_designer.py)
- [`agent_forge/tests/test_consciousness_phenomenology_probe.py`](../../../../tests/test_consciousness_phenomenology_probe.py)

## Strengths

- A directly associated test surface is present in or below this directory.
- Likely validating test files were matched from the surrounding forge test surface.
- The directory exposes importable Python modules rather than only opaque assets.

## Weaknesses / Risks

- The directory has many tracked files but no child-directory decomposition, which may make ownership blur over time.

## Next Steps

- Keep this README synchronized with code and test changes through the managed documentation toolchain.

## References

- Parent README: [`agent_forge/src/agent_forge/consciousness/README.md`](../README.md)
- Forge README: [`agent_forge/README.md`](../../../../README.md)

## Accuracy Contract

- This README is generated from tracked repository structure and conservative code scanning.
- Behavior claims should be backed by tests, routes, or directly linked source files.
<!-- EIDOS:DOCSYS:END -->
