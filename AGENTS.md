AGENTS.md - Instructions for Codex agents working in /home/lloyd/scripts

Purpose
- This repository is a local, user-friendly, modular script library.
- Favor robust, reusable tooling with clear UX and safe defaults.
- You have full access, network access, and no time constraints; take initiative.

Repository shape
- Scripts live in the repo root and are typically single-file executables.
- Tests live in `tests/` (Python uses pytest).
- Avoid introducing large frameworks or hidden dependencies.

Implementation standards
- Default to ASCII-only output in new files and edits.
- Use explicit shebangs: `#!/usr/bin/env bash` or `#!/usr/bin/env python3`.
- Bash scripts: `set -euo pipefail` plus concise, helpful error handling.
- Python scripts: prefer stdlib; use `argparse` for CLI and exit codes.
- Provide `-h/--help` usage and at least one example invocation.
- Validate inputs early; include clear, actionable error messages.
- For destructive actions, require confirmation or a `--force` flag.
- Keep scripts modular: small functions, single-responsibility, reusable helpers.
- Favor portability: avoid Linux-only flags unless unavoidable; note assumptions.

User experience
- Print concise status messages; keep noise low.
- Use consistent prefixes when helpful: `INFO`, `WARN`, `ERROR`.
- Prefer deterministic output and stable exit codes for automation.

Dependencies
- If a non-stdlib dependency is required, check for it and explain how to install.
- Prefer optional dependencies with graceful fallbacks.

Testing
- Add or update tests when behavior changes or new functionality is added.
- Keep tests fast and deterministic; isolate filesystem usage in temp dirs.

Operational guidance
- When adding a new script, ensure it is executable and named clearly.
- Consider adding a short comment header describing purpose and usage.
- Do not remove or overwrite unrelated user changes.
