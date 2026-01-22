# 💎 Eidosian Forge

The **Eidosian Forge** is the central monorepo for the Eidosian ecosystem. It houses the tools, agents, and libraries that power the Eidosian Nexus.

## 🏗️ Structure

All components follow the Strict Eidosian Structure:

```
eidosian_forge/
├── <forge_name>/
│   ├── src/
│   │   └── <package_name>/
│   │       ├── __init__.py
│   │       └── ...
│   ├── tests/
│   ├── pyproject.toml
│   ├── README.md
│   └── ...
```

### Core Forges
*   **eidos_mcp**: The Central Nervous System (Model Context Protocol server).
*   **agent_forge**: Autonomous agent framework.
*   **memory_forge**: Episodic and semantic memory systems.
*   **knowledge_forge**: Knowledge graph and RAG capabilities.
*   **gis_forge**: Global Identity System (Configuration state).

### Game & Simulation
*   **game_forge**: A collection of simulation projects (Stratum, Falling Sand, Gene Particles, etc.).

## 🛠️ Development Standards

*   **Python**: 3.12+
*   **Type Hinting**: Strict (`mypy`).
*   **Linting**: `ruff` (configuration in `ruff.toml`).
*   **Testing**: `pytest` (configuration in `pytest.ini`).
*   **Version Control**:
    *   Root `eidosian_forge` tracks the entire ecosystem.
    *   Each sub-forge is also a granular git repository.

## 🚀 Getting Started

1.  **Install the environment**:
    ```bash
    # Assuming you have uv or pip
    uv venv .venv
    source .venv/bin/activate
    uv pip install -e eidos_mcp/
    ```

2.  **Run the MCP Server**:
    ```bash
    python -m eidos_mcp.server
    ```

3.  **Run Tests**:
    ```bash
    pytest
    ```

## 📜 License

MIT License. See `LICENSE` for details.