# ⚛️ Eidosian Forge v3.14.7

> _"Form follows function; elegance emerges from precision."_

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/) [![License](https://img.shields.io/badge/License-MIT-green)](LICENSE) [![Forge System](https://img.shields.io/badge/Forge-System-8A2BE2)](https://github.com/Ace1928)

```ascii
  ╭───────────────────────────────────╮
  │  TOOLS THAT SHAPE INTELLIGENCE    │
  ╰───────────────────────────────────╯
```

A modular framework of specialized forges for AI interaction, content generation, code manipulation, and knowledge management—where each component functions independently yet cohesively within the ecosystem.

## ⚡ Installation

```bash
git clone https://github.com/Ace1928/eidosian_forge.git
cd eidosian_forge
pip install -e .
```

## 🧠 Core Principles

```ascii
┌────────────────────────────────────────────────────────────┐
│ FORGE DESIGN PHILOSOPHY:                                   │
│                                                            │
│ ↺ Modularity  - Each forge is self-contained               │
│ ↻ Composability - Forges interact seamlessly               │
│ ⟳ Type Safety - Static types prevent errors                │
│ ⟲ Clarity - Self-documenting code, minimal comments        │
└────────────────────────────────────────────────────────────┘
```

## 🛠️ The Forge Ecosystem

```ascii
       ╱▔▔▔▔╲         ╱▔▔▔▔╲         ╱▔▔▔▔╲
      ╱      ╲       ╱      ╲       ╱      ╲
     ╱        ╲     ╱        ╲     ╱        ╲
    ▕    AI    ▏   ▕  Content ▏   ▕   Code   ▏
    ▕  Forges  ▏   ▕  Forges  ▏   ▕  Forges  ▏
     ╲        ╱     ╲        ╱     ╲        ╱
      ╲      ╱       ╲      ╱       ╲      ╱
       ╲▁▁▁▁╱         ╲▁▁▁▁╱         ╲▁▁▁▁╱

       ╱▔▔▔▔╲         ╱▔▔▔▔╲         ╱▔▔▔▔╲
      ╱      ╲       ╱      ╲       ╱      ╲
     ╱        ╲     ╱        ╲     ╱        ╲
    ▕   Data   ▏   ▕  System  ▏   ▕   Viz    ▏
    ▕  Forges  ▏   ▕  Forges  ▏   ▕  Forges  ▏
     ╲        ╱     ╲        ╱     ╲        ╱
      ╲      ╱       ╲      ╱       ╲      ╱
       ╲▁▁▁▁╱         ╲▁▁▁▁╱         ╲▁▁▁▁╱
```

### 🤖 AI Forges

- **agent_forge** — Autonomous agent frameworks with stable decision patterns
- **llm_forge** — LLM interfaces for translating intent to implementation
- **memory_forge** — Persistent context management with retrieval mechanisms
- **ollama_forge** — Local LLM orchestration for offline processing

### 📝 Content Forges

- **article_forge** — Structured document generation with custom formats
- **doc_forge** — Documentation tools that prioritize clarity and completeness
- **lyrics_forge** — Musical content generation with pattern recognition
- **glyph_forge** — Typography tools for consistent visual representation

### 💻 Code Forges

- **code_forge** — Code generation with type safety enforcement
- **refactor_forge** — Code structure transformation preserving behavior
- **repo_forge** — Repository operations with clear audit trails
- **test_forge** — Test generation matching implementation complexity
- **type_forge** — Type inference and annotation management

### 📊 Data Forges

- **archive_forge** — Data storage with integrity verification
- **crawl_forge** — Content retrieval with rate-limiting and respect for robots.txt
- **knowledge_forge** — Information structuring with relationship mapping
- **metadata_forge** — Context enrichment for improved searchability

### 🖥️ System Forges

- **diagnostics_forge** — System state analysis with actionable insights
- **file_forge** — File operations with transactional safety
- **terminal_forge** — Command-line interfaces with intuitive patterns
- **version_forge** — Version control with clean merge resolution
- **mkey_forge** — Key management with encryption and rotation policies

### 🎨 Visualization Forges

- **figlet_forge** — ASCII art generation with consistent rendering
- **gis_forge** — Geographic visualization with coordinate system awareness
- **viz_forge** — Data visualization with perceptual optimization
- **game_forge** — Interactive interfaces with sensible control schemes

### 📡 Communication Forges

- **sms_forge** — SMS utilities with delivery confirmation

## 📚 Usage Examples

```python
from eidosian_forge.code_forge import generate_function
from eidosian_forge.doc_forge import document_code

# Generate a strongly-typed function
function = generate_function(
    name="calculate_risk_score",
    params={"user_data": "Dict[str, Any]", "risk_factors": "List[RiskFactor]"},
    return_type="RiskScore",
    description="Calculates user risk score based on provided factors"
)

# Document it with complete docstrings
documented_function = document_code(function)
```

## 🧪 Local Services

- `eidos-brain` API: `cd eidos-brain && python -m api.server`
- Health checks: `GET /healthz`, `GET /readyz` on `http://localhost:8000`

## 📦 Local GraphRAG Indexing

Use the incremental helper to build a searchable index over local files in batches:

```bash
python scripts/graphrag_local_index.py \
  --root /home/lloyd/graphrag-local-index \
  --scan-root /home/lloyd \
  --batch-size 200 \
  --graphrag-cmd "/home/lloyd/eidosian_forge/graphrag/.venv_tools/bin/poetry run graphrag"
```

## 🧿 Core Design Principles

```ascii
┌──────────────────────────────────────────────────────────┐
│ ʕ•́ᴥ•̀ʔ "Build tools that build better tools, then         │
│         wonder why you didn't start there."              │
└──────────────────────────────────────────────────────────┘
```

1. **Type Integrity** — Static typing prevents more errors than tests catch
2. **Modular Architecture** — Each forge functions independently yet composes seamlessly
3. **Self-Documentation** — Code that explains itself instead of needing comments
4. **Edge Case Handling** — Graceful handling of failures, not silent crashes
5. **Resource Efficiency** — Minimal CPU/memory footprint, maximal capability

## 🔧 Contributing

Contributions welcome. Please maintain type safety, follow naming conventions, and include tests. See CONTRIBUTING.md for details.

## 📜 License

MIT License - Use freely, attribute appropriately, don't blame us for creative applications.

---

> "Complexity hides in what seems simple but isn't." — Eidos
> Maintained by the Eidosian Collective
