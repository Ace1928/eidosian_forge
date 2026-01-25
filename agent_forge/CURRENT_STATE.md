# Current State: agent_forge

**Date**: 2026-01-25
**Status**: Production / Core System
**Version**: E3 (Evolution 3)

## 📊 Metrics

| Metric | Value |
|--------|-------|
| **Python Files** | 50+ |
| **Lines of Code** | ~5,000 |
| **Test Files** | 25 (~100+ tests) |
| **Test Coverage** | ~70% |
| **Dependencies** | pyyaml, rich, typer, sqlite3 |

## 🏗️ Architecture

Agent Forge is the **autonomous agent orchestration system** - the brain that coordinates planning, execution, and reflection for AI agents.

### Core Design

```
┌─────────────────────────────────────────────────────────────┐
│                     AGENT FORGE                              │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────────┐  ┌──────────────────────────────┐   │
│  │    AgentForge     │  │      EidosianAgent           │   │
│  │  (Orchestrator)   │  │    (Main Agent Impl)         │   │
│  └─────────┬─────────┘  └──────────────────────────────┘   │
│            │                                                 │
│  ┌─────────┴──────────────────────────────────────────────┐│
│  │                    CORE SYSTEMS                         ││
│  ├──────────┬──────────┬──────────┬──────────┬──────────┤ │
│  │  State   │  Events  │Scheduler │  Sandbox │ TaskMgr  │ │
│  │ (JSONL)  │(EventBus)│ (Loop)   │(Isolation)│ (Queue)  │ │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘ │
│                                                              │
│  ┌───────────────────┐  ┌──────────────────────────────┐   │
│  │     Planners      │  │      SmolAgentSystem         │   │
│  │   (HTN, YAML)     │  │   (Mini-agent delegation)    │   │
│  └───────────────────┘  └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Key Components

| Component | Purpose | Status |
|-----------|---------|--------|
| **AgentForge** | Top-level orchestrator | ✅ |
| **EidosianAgent** | Main agent implementation | ✅ |
| **TaskManager** | Task queue + dependencies | ✅ |
| **SmolAgentSystem** | Mini-agent delegation | ✅ |
| **State/Journal** | JSONL persistence | ✅ |
| **EventBus** | Append-only events | ✅ |
| **Scheduler** | Loop with backoff | ✅ |
| **Sandbox** | Isolated execution | ✅ |
| **HTNPlanner** | Hierarchical planning | ✅ |
| **CLI Tools** | eidctl, eidosd, eidtop | ✅ |

## 🔌 Features

- **Goal-Oriented** - Goals → Plans → Steps → Runs
- **Task Management** - Priority queue with dependencies
- **Event Sourcing** - Append-only JSONL journal
- **Process Isolation** - Sandboxed execution
- **TUI Monitoring** - Rich terminal interface

## 🔌 Integrations

| Integration | Status |
|-------------|--------|
| **eidos_mcp** | ✅ Transactional ops |
| **code_forge** | ✅ Code analysis |
| **llm_forge** | ✅ Model interface |
| **memory_forge** | ✅ Memory compression |

## 🐛 Known Issues

- Split between `core/` and `src/` needs investigation
- Some tests may require Ollama running

---

**Last Verified**: 2026-01-25