# 💎 Eidosian Forge

> **Brutal Clarity, Unmistakably Eidosian.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](README.md)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/Ace1928/eidosian_forge/workflows/Eidosian%20Universal%20CI/badge.svg)](https://github.com/Ace1928/eidosian_forge/actions/workflows/ci.yml)
[![Lint](https://github.com/Ace1928/eidosian_forge/workflows/Lint%20Code/badge.svg)](https://github.com/Ace1928/eidosian_forge/actions/workflows/lint.yml)
[![Format](https://github.com/Ace1928/eidosian_forge/workflows/Auto%20Format%20Code/badge.svg)](https://github.com/Ace1928/eidosian_forge/actions/workflows/format.yml)

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

### Command Line Interface (CLI)

Eidosian Forge provides a central CLI hub with 10 integrated forge CLIs:

**Central Hub:**
```bash
# Check overall system status
python bin/eidosian status

# List available forges
python bin/eidosian forges

# Route to specific forges
python bin/eidosian memory status
python bin/eidosian knowledge search "eidos"
python bin/eidosian code analyze file.py
python bin/eidosian llm chat "Hello"
python bin/eidosian word define "intelligence"
python bin/eidosian glyph version
python bin/eidosian audit coverage
python bin/eidosian refactor script.py --analyze-only
python bin/eidosian metadata template
```

**Available Forges (10):**

| Forge | Type | Commands |
|-------|------|----------|
| memory | StandardCLI | status, list, search, store, introspect, context, stats, cleanup |
| knowledge | StandardCLI | status, list, search, add, link, path, concepts, unified, stats |
| code | StandardCLI | status, analyze, index, search, ingest, library, stats |
| llm | StandardCLI | status, models, chat, embed, config, test |
| word | StandardCLI | status, lookup, define, related, synsets, graph, build |
| crawl | StandardCLI | status, fetch, extract, robots, tika, cache |
| glyph | Typer | version, interactive, list-commands, bannerize, imagize |
| audit | Typer | coverage, mark, todo |
| refactor | argparse | analyze, refactor (positional file) |
| metadata | argparse | template, validate, version |

**Individual Forge CLIs:**
```bash
# Memory Forge - Tiered memory system
memory-forge status
memory-forge store "Important info" --tier long_term
memory-forge search "important"

# Knowledge Forge - Knowledge graph
knowledge-forge status
knowledge-forge list
knowledge-forge search "architecture"

# Code Forge - Code analysis
code-forge status
code-forge analyze path/to/file.py
code-forge search "function_name"

# LLM Forge - Unified LLM interface
llm-forge status
llm-forge models
llm-forge chat "Explain X"

# Word Forge - Living lexicon
word-forge status
word-forge define "philosophy"
word-forge related "intelligence"

# Glyph Forge - Image to ASCII
glyph-forge version
glyph-forge imagize convert image.png

# Audit Forge - Code audit
audit-forge coverage
audit-forge mark path/to/file.py

# Refactor Forge - Code modularization
refactor-forge script.py --analyze-only
refactor-forge script.py -o ./output

# Metadata Forge - Eidosian metadata
metadata-forge template
metadata-forge validate config.json
```

**Enable Bash Completions:**
```bash
source bin/eidosian-completion.bash
# Add to ~/.bashrc for permanent completions
```

### MCP Server

```bash
python -m eidos_mcp.eidos_mcp_server
```

### Python API

```python
from memory_forge import TieredMemorySystem
from knowledge_forge import KnowledgeForge, KnowledgeMemoryBridge
from code_forge import CodeIndexer

# Memory
memory = TieredMemorySystem()
memory.remember("Important", tier=MemoryTier.LONG_TERM)

# Knowledge
kb = KnowledgeForge("/path/to/kb.json")
kb.add_knowledge("Content", concepts=["topic"])

# Unified search
bridge = KnowledgeMemoryBridge()
results = bridge.unified_search("query")
```

See [QUICKSTART.md](QUICKSTART.md) for detailed examples.

## 🧪 Testing

We use `pytest` with a strict configuration defined in `pytest.ini`.

```bash
# Run all tests
pytest

# Run core tests
pytest -m unit

# Run with coverage
pytest --cov=. --cov-report=html
```

## 🔄 Development & CI/CD

### Automated Workflows

The repository includes comprehensive CI/CD workflows for maintaining code quality:

- **🚀 CI Pipeline** - Runs on every push and PR
  - Multi-version Python testing (3.10, 3.11, 3.12)
  - TypeScript/JavaScript testing
  - Cross-platform testing (Ubuntu + Windows)
  - Documentation building
  - Package validation

- **🎨 Auto-Format** - Formats code automatically
  - Python: Black, isort, Ruff
  - TypeScript/JavaScript: Prettier
  - Auto-commits changes (can be disabled)

- **🔍 Lint** - Comprehensive code quality checks
  - Python: Ruff, Flake8, Pylint, Mypy
  - TypeScript: ESLint, TypeScript compiler
  - Detailed error reporting

### Local Development

Install pre-commit hooks for automatic formatting and linting:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

Run all checks locally before pushing:

```bash
# Run all checks
./scripts/run-checks.sh

# Auto-fix formatting issues
./scripts/run-checks.sh --fix

# Run only Python checks
./scripts/run-checks.sh --python-only

# Run only TypeScript checks
./scripts/run-checks.sh --ts-only
```

See [.github/workflows/README.md](.github/workflows/README.md) for detailed workflow documentation.

## 🤝 Contributing

We follow the **Eidosian Standards**.

1.  **Code Style**: Ruff (configuration in `ruff.toml`). Replaces Black/Isort/Flake8.
2.  **Documentation**: Google-style docstrings, Sphinx-ready.
3.  **Typing**: Strict `mypy` compliance.
4.  **Structure**: All new forges must follow the `src/<package_name>` directory layout.
5.  **Testing**: Use pytest with coverage; maintain high test coverage.
6.  **Pre-commit**: Install and use pre-commit hooks for automatic formatting.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
**Maintained by**: Lloyd Handyside (Neuroforge)
**Contributor**: Eidos
