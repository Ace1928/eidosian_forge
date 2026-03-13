# `.github`

<!-- EIDOS:DOCSYS:START -->
- Contract: `eidos.directory_doc.v1`
- Generated: `2026-03-13T00:19:21Z`
- Path: `.github`

## What It Is

Managed directory documentation for `.github`, generated from tracked files and directory structure.

## Why It Exists

This directory exists to hold the `directory` surface for `.github` and keep that responsibility separate from adjacent forge concerns.

## How It Works

- Tracked files in scope: `22`
- Child directories: `2`
- Tests detected: `False`

## Contents

- [`prompts_old`](./prompts_old/README.md)
- [`workflows`](./workflows/README.md)

## Prominent Files

- [`CODEOWNERS`](./CODEOWNERS)
- [`workflow-config.yml`](./workflow-config.yml)

## Strengths

- Responsibility is split into child directories instead of one flat file heap.

## Weaknesses / Risks

- No directly associated test coverage was detected under the tracked file set for this directory.

## Next Steps

- Add focused tests or point this directory explicitly at its validating test surface.
- Keep this README synchronized with code and test changes through the managed documentation toolchain.

## References

- Forge README: [`.github/README.md`](README.md)

## Accuracy Contract

- This README is generated from tracked repository structure and conservative code scanning.
- Behavior claims should be backed by tests, routes, or directly linked source files.
<!-- EIDOS:DOCSYS:END -->
