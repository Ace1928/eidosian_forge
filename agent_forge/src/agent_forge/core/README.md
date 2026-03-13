# `agent_forge/src/agent_forge/core`

<!-- EIDOS:DOCSYS:START -->
- Contract: `eidos.directory_doc.v1`
- Generated: `2026-03-13T00:19:21Z`
- Path: `agent_forge/src/agent_forge/core`

## What It Is

Managed directory documentation for `agent_forge/src/agent_forge/core`, generated from tracked files and directory structure.

## Why It Exists

This directory exists to hold the `core` surface for `agent_forge` and keep that responsibility separate from adjacent forge concerns.

## How It Works

- Tracked files in scope: `15`
- Child directories: `0`
- Tests detected: `False`
- Python modules: `artifacts, capabilities, config, contracts, db, events, memory, model, os_metrics, sandbox`

## Contents

- No managed child directories detected.

## Prominent Files

- [`__init__.py`](./__init__.py)
- [`artifacts.py`](./artifacts.py)
- [`capabilities.py`](./capabilities.py)
- [`config.py`](./config.py)
- [`contracts.py`](./contracts.py)
- [`db.py`](./db.py)
- [`events.py`](./events.py)
- [`memory.py`](./memory.py)

## Strengths

- The directory exposes importable Python modules rather than only opaque assets.

## Weaknesses / Risks

- No directly associated test coverage was detected under the tracked file set for this directory.
- The directory has many tracked files but no child-directory decomposition, which may make ownership blur over time.

## Next Steps

- Add focused tests or point this directory explicitly at its validating test surface.
- Keep this README synchronized with code and test changes through the managed documentation toolchain.

## References

- Parent README: [`agent_forge/src/agent_forge/README.md`](../README.md)
- Forge README: [`agent_forge/README.md`](../../../README.md)

## Accuracy Contract

- This README is generated from tracked repository structure and conservative code scanning.
- Behavior claims should be backed by tests, routes, or directly linked source files.
<!-- EIDOS:DOCSYS:END -->
