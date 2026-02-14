# 🚀 Eidosian Forge Quick Start Guide

> Get up and running with the Eidosian Forge in 5 minutes.

## Prerequisites

- Python 3.12+
- Ollama (for LLM features)

## Installation

```bash
# 1. Navigate to eidosian_forge
cd /home/lloyd/eidosian_forge

# 2. Activate the virtual environment
source eidosian_venv/bin/activate

# 3. Install core forges (recommended)
pip install -e lib
pip install -e memory_forge
pip install -e knowledge_forge
pip install -e code_forge
pip install -e llm_forge

# 4. Enable bash completions (optional but recommended)
source bin/eidosian-completion.bash

# Add to ~/.bashrc for permanent completions:
echo 'source /home/lloyd/eidosian_forge/bin/eidosian-completion.bash' >> ~/.bashrc
```

## Quick Verification

```bash
# Check overall system status
python bin/eidosian status

# Expected output:
# ╭──────────────────────────────────────╮
# │      EIDOSIAN FORGE v1.0.0        │
# ╰──────────────────────────────────────╯
# 
# System Metrics:
#   MCP Tools:       92
#   Memories:        75
#   Knowledge Nodes: 148
#   Code Elements:   10498
# 
# Forge Status:
#   ✓ memory       operational
#   ✓ knowledge    operational
#   ✓ code         operational
#   ✓ llm          operational
```

## Using Individual Forges

### Memory Forge - Tiered Memory System

```bash
# Check status
memory-forge status

# Store a memory
memory-forge store "Important meeting at 3pm" --tier working --tags meeting

# Search memories
memory-forge search "meeting"

# Get auto-context suggestions
memory-forge context "What meetings do I have?"
```

### Knowledge Forge - Knowledge Graph

```bash
# Check status
knowledge-forge status

# List knowledge nodes
knowledge-forge list

# Search knowledge
knowledge-forge search "architecture"

# Add new knowledge
knowledge-forge add "Eidosian Forge uses Python 3.12" --concepts python eidos

# Find paths between concepts
knowledge-forge path <node_a_id> <node_b_id>
```

### Code Forge - Code Analysis

```bash
# Check status
code-forge status

# Analyze a Python file
code-forge analyze path/to/file.py

# Search indexed code
code-forge search "TieredMemory"

# View code statistics
code-forge stats
```

### LLM Forge - Unified LLM Interface

```bash
# Check status (shows Ollama connection)
llm-forge status

# List available models
llm-forge models

# Chat with model
llm-forge chat "Explain the Eidosian philosophy in one sentence"

# Test connection
llm-forge test
```

## Using the Central Hub

The `eidosian` command provides unified access to all forges:

```bash
# System status
python bin/eidosian status

# Route to forges
python bin/eidosian memory status
python bin/eidosian knowledge search "eidos"
python bin/eidosian code analyze file.py
python bin/eidosian llm chat "Hello"

# List available forges
python bin/eidosian forges
```

## Python API Quick Examples

### Memory Forge

```python
from memory_forge import TieredMemorySystem, MemoryTier, MemoryNamespace

memory = TieredMemorySystem()

# Store
memory_id = memory.remember(
    content="Important insight about X",
    tier=MemoryTier.LONG_TERM,
    namespace=MemoryNamespace.KNOWLEDGE,
    tags={"insight", "important"},
)

# Recall
results = memory.recall("insight", limit=5)
```

### Knowledge Forge

```python
from knowledge_forge import KnowledgeForge

kb = KnowledgeForge("/home/lloyd/eidosian_forge/data/kb.json")

# Add knowledge
node = kb.add_knowledge(
    content="Important concept",
    concepts=["category"],
    tags=["tag1"],
)

# Search
results = kb.search("concept")
```

### Code Forge

```python
from code_forge import CodeIndexer

indexer = CodeIndexer()

# Search code elements
for elem in indexer.elements.values():
    if "memory" in elem.name.lower():
        print(f"{elem.element_type}: {elem.qualified_name}")
```

## MCP Server

Start the MCP server for tool integration:

```bash
cd /home/lloyd/eidosian_forge/eidos_mcp
source ../eidosian_venv/bin/activate
python -m eidos_mcp.eidos_mcp_server
```

## Data Locations

| Data | Location |
|------|----------|
| Memory | `/home/lloyd/eidosian_forge/data/memory/` |
| Knowledge | `/home/lloyd/eidosian_forge/data/kb.json` |
| Code Index | `/home/lloyd/eidosian_forge/data/code_index.json` |
| Model Config | `/home/lloyd/eidosian_forge/data/model_config.json` |

## Troubleshooting

### Import Errors
```bash
export PYTHONPATH="${PYTHONPATH}:/home/lloyd/eidosian_forge/lib"
```

### `eidosian_core` Not Found
```bash
cd /home/lloyd/eidosian_forge
source eidosian_venv/bin/activate
pip install -e lib
```

### Ollama Not Running
```bash
ollama serve  # Start Ollama
ollama list   # Check available models
```

### Missing Models
```bash
ollama pull phi3:mini         # Inference model
ollama pull nomic-embed-text  # Embedding model
```

## Next Steps

- Read individual forge INSTALL.md files for detailed documentation
- Explore the MCP tools via `eidos_mcp`
- Check `/home/lloyd/eidosian_forge/docs/FORGE_ARCHITECTURE.md` for system architecture
