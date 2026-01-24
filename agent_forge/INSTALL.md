# Agent Forge Installation

## Quick Install

```bash
# From eidosian_forge root
cd agent_forge
pip install -e .

# Or install with all dependencies
pip install -e ".[dev]"
```

## CLI Tools

Agent Forge provides three CLI tools:

1. **eidctl** - State management and inspection
   ```bash
   eidctl state snapshot
   eidctl journal list
   eidctl goals add "My Goal"
   ```

2. **eidosd** - Daemon scheduler
   ```bash
   eidosd --once    # Single beat
   eidosd --loop    # Continuous scheduler
   ```

3. **eidtop** - Real-time metrics TUI
   ```bash
   eidtop
   ```

## Via Central Hub

```bash
# Through eidosian command
eidosian agent state snapshot
eidosian agent-daemon --once
eidosian agent-top
```

## Bash Completions

```bash
source agent_forge/completions/agent-forge.bash
```
