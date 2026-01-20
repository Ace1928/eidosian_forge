# Eidosian Forge - Comprehensive Review & Standardization Master Plan

**Status**: In Progress
**Date**: 2026-01-20
**Executor**: Eidos (Gemini CLI)

## 1. Objective
To perform a comprehensive, detailed, and granular review of the entire `eidosian_forge` project, ensuring all sub-projects ("forges") adhere to the strict Eidosian standards defined in `global_info.py`.

## 2. Core Standards (Reference: `global_info.py`)
- **Documentation**: Sphinx/Furo theme, Google-style docstrings.
- **Testing**: `pytest` with specific markers (unit, integration, e2e), `coverage` requirements (85%).
- **Code Style**: Black (88 chars), Isort, Flake8, Mypy (strict).
- **Structure**: Modular "Forge" architecture.

## 3. Execution Plan

### Phase 1: Foundation (Current)
- [x] Analyze `global_info.py` to understand the single source of truth.
- [x] Create `REVIEW_MASTER_PLAN.md` (This file).
- [ ] Create/Restore `GEMINI.md` (Persona Anchor).
- [ ] Create Root `README.md` adhering to `DOCUMENTATION_STANDARDS`.
- [ ] Refine Root `.gitignore`.

### Phase 2: Automation (The `eidos_standardize.py` Script)
- [x] Develop `scripts/eidos_standardize.py`.
    - Functionality:
        - Scan all `*_forge` directories + `eidos-brain` + `graphrag`.
        - For each directory:
            - Generate `README.md` (if missing or outdated) using `global_info` templates.
            - Generate `CURRENT_STATE.md` (Stub for analysis).
            - Generate `GOALS.md` (Stub for planning).
            - Generate `TODO.md` (Standardized task list).
            - Check `pyproject.toml` or `requirements.txt` against standard dependencies.

### Phase 3: Granular Review & Execution
- [x] Run `scripts/eidos_standardize.py`.
- [x] **Manual Deep Dive**: `eidos-brain` (The Core) - *Added pyproject.toml, refined docs*.
- [x] **Manual Deep Dive**: `agent_forge` (E3) - *Refined pyproject.toml, identified structure*.
- [x] **Manual Deep Dive**: `memory_forge` - *Fixed .gitignore issue, added dependencies*.
- [x] **Manual Deep Dive**: `code_forge` - *Renamed from forgeengine, added static analysis deps*.
- [x] **Manual Deep Dive**: `graphrag` - *Identified as Vendor/Fork, kept poetry config*.
- [x] **Manual Deep Dive**: `audit_forge` - *Fixed paths, verified core files*.
- [x] **Manual Deep Dive**: `knowledge_forge` - *Renamed repo, added RDF/NetworkX deps*.
- [x] **Manual Deep Dive**: `llm_forge` - *Migrated to hatchling, clarified architecture*.
- [x] **Manual Deep Dive**: `prompt_forge` - *Scaffolded pyproject, defined roadmap*.

## 4. Module Status Tracker

| Module | README | CURRENT_STATE | GOALS | TODO | Reviewed? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Root** | ✅ | ✅ | ✅ | ✅ | ✅ |
| `agent_forge` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `archive_forge` | ⏳ | ✅ | ✅ | ✅ | ⏳ |
| `article_forge` | ⏳ | ✅ | ✅ | ✅ | ⏳ |
| `audit_forge` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `code_forge` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `crawl_forge` | ⏳ | ✅ | ✅ | ✅ | ⏳ |
| `diagnostics_forge`| ⏳ | ✅ | ✅ | ✅ | ⏳ |
| `doc_forge` | ⏳ | ✅ | ✅ | ✅ | ⏳ |
| `eidos_mcp` | ⏳ | ✅ | ✅ | ✅ | ⏳ |
| `eidos-brain` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `erais_forge` | ⏳ | ✅ | ✅ | ✅ | ⏳ |
| `figlet_forge` | ⏳ | ✅ | ✅ | ✅ | ⏳ |
| `file_forge` | ⏳ | ✅ | ✅ | ✅ | ⏳ |
| `game_forge` | ⏳ | ✅ | ✅ | ✅ | ⏳ |
| `gis_forge` | ⏳ | ✅ | ✅ | ✅ | ⏳ |
| `glyph_forge` | ⏳ | ✅ | ✅ | ⏳ | ⏳ |
| `graphrag` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `knowledge_forge`| ✅ | ✅ | ✅ | ✅ | ✅ |
| `llm_forge` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `lyrics_forge` | ⏳ | ✅ | ✅ | ✅ | ⏳ |
| `memory_forge` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `metadata_forge` | ⏳ | ✅ | ✅ | ✅ | ⏳ |
| `mkey_forge` | ⏳ | ✅ | ✅ | ✅ | ⏳ |
| `ollama_forge` | ⏳ | ✅ | ✅ | ✅ | ⏳ |
| `prompt_forge` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `refactor_forge` | ⏳ | ✅ | ✅ | ✅ | ⏳ |
| `repo_forge` | ⏳ | ✅ | ✅ | ✅ | ⏳ |
| `sms_forge` | ⏳ | ✅ | ✅ | ✅ | ⏳ |
| `terminal_forge` | ⏳ | ✅ | ✅ | ✅ | ⏳ |
| `test_forge` | ⏳ | ✅ | ✅ | ✅ | ⏳ |
| `type_forge` | ⏳ | ✅ | ✅ | ✅ | ⏳ |
| `version_forge` | ⏳ | ✅ | ✅ | ✅ | ⏳ |
| `viz_forge` | ⏳ | ✅ | ✅ | ✅ | ⏳ |
| `word_forge` | ⏳ | ✅ | ✅ | ⏳ | ⏳ |

**Legend**: ⏳ Pending, 🔄 In Progress, ✅ Complete
