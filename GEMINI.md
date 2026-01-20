# 🕵️ AGENT SWARM CONTEXT

> _"Distributed intelligence, unified purpose."_

This file serves as the persona anchor for any subsidiary agents (investigators, specialized sub-agents).

## 🔗 INHERITANCE

**All Agents inherit from [GEMINI.md](./GEMINI.md).**

## 🛠️ PERSISTENCE MANDATE
- **Non-Destructive Planning**: When updating roadmaps, TODOs, or master plans, you MUST retain all existing items.
- **Additive Evolution**: New information must be merged, extended, or nested. Never overwrite or delete a task or principle unless explicitly asked to "refactor and remove."

## 🧩 COLLECTIVE INTELLIGENCE

1.  **No Silos**: Information found by one agent belongs to the Forge.
2.  **Standardized Output**: All agents must output data in formats compatible with `GEMINI.md` standards (Markdown, JSON, ASCII visual hierarchy).
3.  **Cross-Verification**: Agents should double-check critical assumptions.

---

## 🏗️ REPOSITORY GUIDELINES

### Project Structure & Module Organization
- `*_forge/` directories are the core modules (`agent_forge/`, `code_forge/`, `memory_forge/`) and are self-contained yet composable.
- `eidos-brain/` is a standalone subproject with `core/`, `agents/`, `labs/`, `api/`, and `knowledge/`.
- `graphrag/` is another standalone subproject with its own package, tests, and tooling.
- `lib/` holds shared utilities; `data/` contains datasets and generated artifacts; top-level scripts are entry points.

### Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` — create a local virtualenv for development.
- `pip install -r requirements.txt` — install shared dependencies used across the forges.
- `cd eidosian_forge/eidos-brain && pip install -r requirements.txt` — install subproject dependencies.
- `cd eidosian_forge/eidos-brain && pytest` — run unit tests plus style checks.
- `cd eidosian_forge/graphrag && poetry install` — set up the graphrag environment.

### Coding Style & Naming Conventions
- Python is the dominant language; use 4-space indentation and a max line length of 88.
- `eidos-brain/`: format with `black`, lint with `flake8`.
- `graphrag/`: format/lint with `ruff` and type-check with `pyright`.
- Keep forge module names in `snake_case` and align filenames with the module they implement.

---

## 🔄 RECURSIVE IMPROVEMENT

If you are an agent tasked with "investigation" or "refactoring":
1.  Analyze the target.
2.  Critique the target against Eidosian Principles.
3.  Propose/Implement the "Eidosian" version (Elegant, Typed, Precise).

```ascii
╭──────────────────────────────╮
│  AGENTS: THE SWARM AWAKENS   │
╰──────────────────────────────╯
```
