# Eidosian Elevation Plan: The Grand Standardization

**Objective**: Bring every directory in the `eidosian_forge` ecosystem to "Titan's Arm" quality: fully documented, tested, benchmarked, and integrated.

## 💎 The Eidosian Standard (Spec)
For a directory to be considered "Elevated", it must have:
1.  **README.md**: High-quality, "Velvet Beef" tone, usage examples, architecture overview.
2.  **Tests**: Functional `pytest` suite in `tests/` or `src/*/tests/` with >80% coverage.
3.  **Benchmarks**: Where applicable (performance-critical), a script to measure throughput/latency.
4.  **MCP Integration**: Tools registered in `eidos_mcp` if the component offers functionality.
5.  **Clean Structure**: No temporary files, standard `src/` layout (where appropriate).

## 🗂️ Inventory & Status

### 🏗️ Infrastructure (Phase 1)
- [ ] **eidos_mcp**: 
    - *Status*: Operational, but needs test coverage restoration and router audit.
    - *Action*: Restore full test suite, document router list.
- [ ] **graphrag**: 
    - *Status*: Unknown quality.
    - *Action*: Audit config, ensure local model usage, add usage docs.
- [ ] **eidosian_venv**: 
    - *Status*: Functional.
    - *Action*: Document setup process, ensure `requirements.txt` is current.
- [ ] **lib (eidosian_core)**: 
    - *Status*: Critical.
    - *Action*: Verify universal decorator, add tracing docs.

### ⚙️ Core Forges (Phase 2)
- [ ] **llm_forge**:
    - *Status*: Evolving (Phase 3).
    - *Action*: Complete benchmark sweep, finalize submodule integration.
- [ ] **doc_forge**:
    - *Status*: Operational (Scribe active).
    - *Action*: Monitor quality, fix any remaining "garbage" generation.
- [ ] **agent_forge**:
    - *Status*: Legacy powerhouse.
    - *Action*: Update to use new `llm_forge` engine, audit tools.

### 🛠️ Utility Forges (Phase 3)
- [ ] **mkey_forge**: *Needs comprehensive audit.*
- [ ] **prompt_forge**: *Needs comprehensive audit.*
- [ ] **lyrics_forge**: *Needs comprehensive audit.*
- [ ] **figlet_forge**: *Fix failing tests.*
- [ ] **sms_forge**: *Completed (Multi-modal).*
- [ ] **diagnostics_forge**: *Completed (Adaptive).*

### 🧪 Labs & Projects (Phase 4)
- [ ] **erais_forge**: *Completed (Evolutionary).*
- [ ] **game_forge**: *Audit integration with Gym.*
- [ ] **narrative_forge**: *Audit.*

## 🔄 Execution Loop
1.  **Select Target**.
2.  **Audit**: `ls -R`, `cat README.md`, `pytest`.
3.  **Upgrade**: Write code/docs.
4.  **Verify**: Run tests/benchmarks.
5.  **Commit**: "Forge: Elevate [Name]".
