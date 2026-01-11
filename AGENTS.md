# Repository Guidelines

## Project Structure & Module Organization
- `*_forge/` directories are the core modules (`agent_forge/`, `code_forge/`, `memory_forge/`) and are self-contained yet composable.
- `eidos-brain/` is a standalone subproject with `core/`, `agents/`, `labs/`, `api/`, and `knowledge/`.
- `graphrag/` is another standalone subproject with its own package, tests, and tooling.
- `lib/` holds shared utilities; `data/` contains datasets and generated artifacts; top-level scripts (like `run_gene_particles.py`) are entry points.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` — create a local virtualenv for development.
- `pip install -r requirements.txt` — install shared dependencies used across the forges.
- `cd eidos-brain && pip install -r requirements.txt` — install the eidos-brain subproject dependencies.
- `cd eidos-brain && pytest` — run unit tests plus style checks (black + flake8).
- `cd eidos-brain && python labs/tutorial_app.py` — launch the tutorial app.
- `cd eidos-brain && python -m api.server` — run the API locally (`/healthz`, `/readyz`).
- `cd graphrag && poetry install` — set up the graphrag environment.
- `cd graphrag && poetry run poe test` — run graphrag tests.
- `cd graphrag && poetry run poe check` — run ruff + pyright checks.
- `python scripts/graphrag_local_index.py --root <project> --scan-root <path>` — index local files in batches.

## Coding Style & Naming Conventions
- Python is the dominant language; use 4-space indentation and a max line length of 88.
- `eidos-brain/`: format with `black`, lint with `flake8`.
- `graphrag/`: format/lint with `ruff` and type-check with `pyright` per `pyproject.toml`.
- Keep forge module names in `snake_case` and align filenames with the module they implement (like `memory_store.py` in `memory_forge/`).

## Testing Guidelines
- Use `pytest` across subprojects; tests live in `eidos-brain/tests/` and `graphrag/tests/`.
- Name tests `test_*.py` with focused, single-purpose cases.
- If you add a new forge capability, add at least one unit test in the nearest subproject test suite.

## Commit & Pull Request Guidelines
- Commit messages use imperative, sentence-case verbs (for example, “Add …”, “Update …”, “Remove …”).
- PRs should include a clear description, testing notes, and linked issues if applicable.
- For graphrag changes, follow `graphrag/CONTRIBUTING.md` and versioning notes in `graphrag/DEVELOPING.md`.

## Configuration & Secrets
- Store local settings in `.env` files where supported (not committed).
- Avoid committing large generated files or credentials; document new required env vars in the relevant subproject README.
