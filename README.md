# 💎 Eidosian Forge

> **Brutal Clarity, Unmistakably Eidosian.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](README.md)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

## 📋 Overview

The **Eidosian Forge** is a monolithic repository containing a suite of specialized "Forges" (modules) designed to construct the **Eidosian Intelligence System**. It houses the tools, agents, libraries, and the central nervous system that power the Eidosian Nexus.

It serves as the central foundry for developing distributed AI agents, knowledge graphs, code analysis tools, and memory systems. The project follows a strict "Forge" architecture where each `*_forge` directory represents a composable, self-contained domain of capability, all governed by unified standards.

## 🏗️ Project Structure

The repository is organized into specialized domains following a strict `src`-based layout:

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

| Forge | Description |
| :--- | :--- |
| **`eidos_mcp`** | The Central Nervous System (Model Context Protocol server). Replaces legacy `eidos_brain`. |
| **`agent_forge`** | Toolkit for building, managing, and deploying autonomous agents. |
| **`memory_forge`** | Long-term memory, semantic storage, and context management. |
| **`knowledge_forge`** | Knowledge graph construction and ontology management. |
| **`gis_forge`** | Global Identity System (Configuration state). |
| **`code_forge`** | Source code analysis, generation, and refactoring tools. |
| **`llm_forge`** | Interfaces and abstractions for Large Language Models. |
| **`prompt_forge`** | Prompt engineering, templating, and management. |
| **`game_forge`** | A collection of simulation projects (Stratum, Falling Sand, Gene Particles, etc.). |

*See `REVIEW_MASTER_PLAN.md` for a full list of active modules.*

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.12+
- **Node.js**: 18+ (for web components)
- **Rust**: 1.70+ (for high-performance extensions)

### Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Neuroforge/eidosian_forge.git
    cd eidosian_forge
    ```

2.  **Set up the Environment**:
    ```bash
    # Activate the canonical Eidosian virtual environment
    # Ensure you have moved/created it at eidosian_forge/eidosian_venv
    source eidosian_venv/bin/activate
    
    # Install the MCP server and core forges in editable mode using standard pip
    pip install -e eidos_mcp/
    ```

3.  **Bootstrap a Specific Forge** (e.g., `agent_forge`):
    ```bash
    cd agent_forge
    pip install -e .
    ```

## 🛠️ Usage

This repository is a collection of tools. Typical usage involves running the MCP server or specific CLI entry points.

**Example: Running the Eidosian MCP Server**
```bash
python -m eidos_mcp.server
```

**Example: Agent Task (Theoretical)**
```bash
python -m agent_forge.cli --task "Analyze codebase"
```

## 🧪 Testing

We use `pytest` with a strict configuration defined in `pytest.ini`.

```bash
# Run all tests
pytest

# Run core tests
pytest -m unit
```

## 🤝 Contributing

We follow the **Eidosian Standards**.

1.  **Code Style**: Ruff (configuration in `ruff.toml`). Replaces Black/Isort/Flake8.
2.  **Documentation**: Google-style docstrings, Sphinx-ready.
3.  **Typing**: Strict `mypy` compliance.
4.  **Structure**: All new forges must follow the `src/<package_name>` directory layout.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
**Maintained by**: Lloyd Handyside (Neuroforge)
**Contributor**: Eidos
