# TODO: agent_forge

## 🚨 High Priority
- [ ] **Standardize**: Ensure `bin/` scripts work with the new `pyproject.toml` entry points.
- [ ] **Audit**: Investigate `src/` vs `core/`.
    - *Hypothesis*: `core` is the engine, `src` might be legacy or specific implementations.

## 🟡 Medium Priority
- [ ] **Docs**: Document the "Journal" architecture.
- [ ] **Features**: Enhance `eidtop` (TUI).

## 🟢 Low Priority
- [ ] **Cleanup**: Remove `uv.lock` if we are moving to standard pip/venv, or fully embrace `uv`.