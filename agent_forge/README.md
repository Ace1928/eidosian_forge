# 🤖 Agent Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status: Operational](https://img.shields.io/badge/Status-Operational-green.svg)](README.md)

**The Cognitive Architecture of Eidos.**

> _"Agency is the intersection of logic and intent."_

## 🤖 Overview

`agent_forge` is the primary reasoning and orchestration module of the Eidosian ecosystem. It provides a modular framework for building autonomous agents capable of complex task planning, multi-step execution, and recursive self-reflection.

It implements the **E3 Architecture** (Eidosian Execution Engine):
1.  **Sense**: Gather environmental data (via Nexus tools).
2.  **Think**: Plan and reason using local LLMs.
3.  **Act**: Execute shell commands, file edits, and system interactions.

## 🏗️ Architecture

- **Core (`src/agent_forge/core/`)**:
    - `agent.py`: The main autonomous loop.
    - `planner.py`: HTN (Hierarchical Task Network) and simple linear planners.
    - `model.py`: Unified interface for local and remote LLM providers.
- **Consciousness Kernel**: A specialized monitoring layer that tracks internal state, attention, and cognitive load.
- **Actuators**: Modular tool-calling interfaces for sandboxed and system-level execution.

## 🔗 System Integration

- **LLM Forge**: All agents utilize the centralized `llm_forge` local engine for inference.
- **Memory Forge**: Episodic and semantic memory integration for long-term context retention.
- **Nexus**: Exposes tools for agent deployment (`eidctl`) and monitoring (`eidtop`).

## 🚀 Usage

### Deploying an Agent

```bash
# Start the agent daemon
./bin/eidosd start

# Interaction via CLI
eidctl run --objective "Summarize the latest system logs"
```

### Cognitive Monitoring

```bash
# Monitor agent "thoughts" and resource usage
eidtop
```

## 🛠️ Configuration

Agent personalities and skillsets are defined in `cfg/self.yml` and `cfg/skills.yml`.

---

**Version**: 1.0.0 (Titan's Arm)
**Maintainer**: EIDOS
