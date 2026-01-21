# Eidosian Forge

> **Brutal Clarity, Unmistakably Eidosian.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📋 Overview

The **Eidosian Forge** is a monolithic repository containing a suite of specialized "Forges" (modules) designed to construct the **Eidosian Intelligence System**. It serves as the central foundry for developing distributed AI agents, knowledge graphs, code analysis tools, and memory systems.

The project follows a strict "Forge" architecture where each `*_forge` directory represents a composable, self-contained domain of capability, all governed by the unified standards defined in `global_info.py`.

## 🏗️ Project Structure

The repository is organized into specialized domains:

| Forge | Description |
| :--- | :--- |
| **`eidos_brain`** | The central cognitive core, managing agent orchestration and reasoning. |
| **`agent_forge`** | Toolkit for building, managing, and deploying autonomous agents. |
| **`memory_forge`** | Long-term memory, semantic storage, and context management. |
| **`code_forge`** | Source code analysis, generation, and refactoring tools. |
| **`knowledge_forge`** | Knowledge graph construction and ontology management. |
| **`graphrag`** | Graph-based Retrieval Augmented Generation implementation. |
| **`llm_forge`** | Interfaces and abstractions for Large Language Models. |
| **`prompt_forge`** | Prompt engineering, templating, and management. |
| **`tool_forge`** | (Implied) Tool definitions and Actuators. |

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
    # Create and activate virtual environment
    python3.12 -m venv .venv
    source .venv/bin/activate
    
    # Install shared dependencies
    pip install -r requirements.txt
    ```

3.  **Bootstrap a Specific Forge** (e.g., `eidos_brain`):
    ```bash
    cd eidos_brain
    pip install -r requirements.txt
    ```

## 🛠️ Usage

This repository is a collection of tools. Typical usage involves importing specific forges or running their CLI entry points.

**Example: Running the Global Info System**
```bash
python global_info.py
```

**Example: Agent Task (Theoretical)**
```bash
python agent_forge/bin/run_agent.py --task "Analyze codebase"
```

## 🧪 Testing

We use `pytest` with a strict configuration.

```bash
# Run all tests
pytest

# Run core tests
pytest -m unit
```

## 🤝 Contributing

We follow the **Eidosian Standards**.

1.  **Code Style**: Black (88 chars), Isort, Flake8.
2.  **Documentation**: Google-style docstrings, Sphinx-ready.
3.  **Typing**: Strict `mypy` compliance.

See `global_info.py` for the definitive configuration source.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
**Maintained by**: Lloyd Handyside (Neuroforge)
**Contributor**: Eidos
