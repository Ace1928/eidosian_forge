"""Agent Forge package exports with defensive import behavior."""

try:
    from .agent_core import AgentForge, Task, Goal
except Exception:  # pragma: no cover - allows partial subsystem startup
    AgentForge = None  # type: ignore[assignment]
    Task = None  # type: ignore[assignment]
    Goal = None  # type: ignore[assignment]

__all__ = ["AgentForge", "Task", "Goal"]
