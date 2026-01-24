# 🏗️ Eidosian Forge Architecture

> **Brutal Clarity, Unmistakably Eidosian.**

This document describes the architecture of the Eidosian Forge system - a modular cognitive architecture designed for autonomous AI agents.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      EIDOSIAN FORGE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │   Central   │    │    MCP      │    │   Unified   │        │
│   │    CLI      │────│   Server    │────│   Config    │        │
│   │  (eidosian) │    │ (eidos_mcp) │    │ (data/*.json)│       │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│          │                  │                                    │
│          ▼                  ▼                                    │
│   ┌─────────────────────────────────────────────────────┐      │
│   │                 FORGE LAYER                          │      │
│   ├──────────┬──────────┬──────────┬──────────┬────────┤      │
│   │ memory   │knowledge │  code    │   llm    │  word  │      │
│   │ _forge   │ _forge   │ _forge   │ _forge   │ _forge │      │
│   ├──────────┼──────────┼──────────┼──────────┼────────┤      │
│   │  crawl   │  glyph   │  audit   │ refactor │terminal│      │
│   │ _forge   │ _forge   │ _forge   │ _forge   │ _forge │      │
│   └──────────┴──────────┴──────────┴──────────┴────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Central CLI (`bin/eidosian`)

The central command hub that routes commands to individual forges.

```bash
eidosian status           # System-wide status
eidosian forges           # List available forges
eidosian memory status    # Route to memory_forge
eidosian knowledge search # Route to knowledge_forge
```

**Architecture:**
- Uses `FORGE_REGISTRY` dict to map subcommands to forge modules
- Supports three CLI types: StandardCLI, Typer, argparse
- Handles --help routing specially to pass to forges

### 2. MCP Server (`eidos_mcp`)

Model Context Protocol server exposing forge capabilities as tools.

```python
from eidos_mcp import EidosMCPServer
server = EidosMCPServer()
# 92 tools registered across routers
```

**Routers:**
- `tiered_memory` - 19 memory tools
- `knowledge` - 14 knowledge graph tools
- `code` - 5 code analysis tools
- `system` - 12 system tools
- `tika` - 8 document extraction tools
- `audit` - 5 audit tools

### 3. StandardCLI Framework (`lib/cli/`)

Reusable base class for building forge CLIs.

```python
from cli import StandardCLI, CommandResult

class MyForgeCLI(StandardCLI):
    name = "my_forge"
    description = "My forge description"
    version = "1.0.0"
    
    def register_commands(self, subparsers):
        # Add forge-specific commands
        pass
    
    def cmd_status(self, args) -> CommandResult:
        return CommandResult(True, "OK", {"status": "operational"})
```

**Features:**
- Global args: --json, --no-color, --quiet, --version
- Standard commands: status, info, integrations
- Bash completion generation
- Forge integration detection

## Forge Architecture

### Directory Structure

Each forge follows a standard layout:

```
<forge>_forge/
├── src/
│   └── <forge>_forge/
│       ├── __init__.py
│       ├── cli.py
│       ├── core/
│       └── ...
├── tests/
├── completions/
│   └── <forge>-forge.bash
├── pyproject.toml
├── INSTALL.md
└── README.md
```

### Forge Categories

#### Tier A: Core Infrastructure
| Forge | Purpose | Key Classes |
|-------|---------|-------------|
| memory_forge | Tiered memory system | `TieredMemorySystem` |
| knowledge_forge | Knowledge graph | `KnowledgeForge`, `KnowledgeMemoryBridge` |
| code_forge | Code analysis | `CodeIndexer`, `CodeAnalyzer` |
| llm_forge | LLM interface | `OllamaClient` |
| word_forge | Living lexicon | `GraphManager`, `SemanticNetwork` |
| crawl_forge | Web extraction | `TikaExtractor` |

#### Tier B: Processing
| Forge | Purpose | Key Classes |
|-------|---------|-------------|
| glyph_forge | Image to ASCII | `GlyphForgeAPI` |
| audit_forge | Code audit | `AuditForge` |
| refactor_forge | Code modularization | `analyze_code`, `refactor` |
| metadata_forge | Metadata management | `create_metadata_template` |
| terminal_forge | Terminal styling | `Color`, `Theme`, `Banner` |

#### Tier C: Specialized
| Forge | Purpose |
|-------|---------|
| figlet_forge | ASCII text banners |
| agent_forge | Autonomous agents |
| doc_forge | Documentation |
| game_forge | Simulations |

## Data Architecture

### Memory System

```
┌─────────────────────────────────────────┐
│           TIERED MEMORY                  │
├─────────────────────────────────────────┤
│  SHORT_TERM    (TTL: 1 hour)            │
│  WORKING       (TTL: 24 hours)          │
│  LONG_TERM     (Persistent)             │
│  SELF          (EIDOS identity)         │
│  USER          (User preferences)       │
├─────────────────────────────────────────┤
│  Namespaces: EIDOS, USER, TASK,         │
│              KNOWLEDGE, CODE, CONVERSATION│
└─────────────────────────────────────────┘
```

**Storage:** `data/memory/tiered_memory.json`

### Knowledge Graph

```
┌─────────────────────────────────────────┐
│         KNOWLEDGE FORGE                  │
├─────────────────────────────────────────┤
│  Nodes: 148+                            │
│  Concepts: Tagged and linked            │
│  Embeddings: nomic-embed-text (768 dim) │
├─────────────────────────────────────────┤
│  Bridge: KnowledgeMemoryBridge          │
│  Unified search across memory + KB      │
└─────────────────────────────────────────┘
```

**Storage:** `data/kb.json`

### Code Index

```
┌─────────────────────────────────────────┐
│           CODE INDEXER                   │
├─────────────────────────────────────────┤
│  Elements: 10,498                        │
│  Files: 1,379 Python files              │
│  Types: functions, classes, methods      │
├─────────────────────────────────────────┤
│  AST-based analysis                      │
│  Semantic search (planned)               │
└─────────────────────────────────────────┘
```

**Storage:** `data/code_index.json`

## Model Configuration

All forges use unified model configuration:

```json
{
  "inference": {
    "model": "phi3:mini",
    "provider": "ollama"
  },
  "embedding": {
    "model": "nomic-embed-text",
    "dimensions": 768
  }
}
```

**Config:** `data/model_config.json`

## CLI Types

### StandardCLI (7 forges)
- memory, knowledge, code, llm, word, crawl, terminal
- Uses `lib/cli/__init__.py` base class
- Consistent interface and output formatting

### Typer (2 forges)
- glyph, audit
- Rich terminal output
- Built-in completion

### Argparse (2 forges)
- refactor, metadata
- Standard Python argparse

## Integration Patterns

### Cross-Forge Communication

```python
# Memory ↔ Knowledge Bridge
from knowledge_forge import KnowledgeMemoryBridge

bridge = KnowledgeMemoryBridge()
results = bridge.unified_search("query", limit=10)
```

### Forge Detection

```python
from cli import ForgeDetector

available = ForgeDetector.available_forges()
# ['memory_forge', 'knowledge_forge', ...]
```

### Enhanced Capabilities

When multiple forges are present, capabilities are enhanced:

- memory_forge + knowledge_forge → Unified search
- crawl_forge + knowledge_forge → Auto-ingestion
- code_forge + knowledge_forge → Semantic code search

## File Locations

| Component | Path |
|-----------|------|
| Central CLI | `bin/eidosian` |
| Bash completions | `bin/eidosian-completion.bash` |
| CLI framework | `lib/cli/__init__.py` |
| MCP server | `eidos_mcp/src/eidos_mcp/` |
| Memory data | `data/memory/` |
| Knowledge data | `data/kb.json` |
| Code index | `data/code_index.json` |
| Model config | `data/model_config.json` |

## Extending the System

### Adding a New Forge

1. Create forge directory structure
2. Implement StandardCLI subclass
3. Add to `FORGE_REGISTRY` in `bin/eidosian`
4. Create bash completions
5. Create INSTALL.md
6. Update eidosian-completion.bash

### Adding MCP Tools

1. Create router in `eidos_mcp/src/eidos_mcp/routers/`
2. Register in `plugins/core.py`
3. Document tools

---

**Version:** 1.0.0
**Last Updated:** 2026-01-24
**Maintained by:** EIDOS
