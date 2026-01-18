"""
Eidosian Forge agent system.

This module provides the core agent implementation, task management,
specialized agents, and interaction capabilities.
"""

from .agent import EidosianAgent
from .prompt_templates import (
    AGENT_DECISION_TEMPLATE,
    AGENT_PLAN_TEMPLATE,
    AGENT_REFLECTION_TEMPLATE,
    COLLABORATION_TEMPLATE,
    SMOL_AGENT_TASK_TEMPLATE,
)
from .smol_agents import SmolAgentSystem
from .task_manager import TaskManager

__all__ = [
    "EidosianAgent",
    "TaskManager",
    "SmolAgentSystem",
    "AGENT_REFLECTION_TEMPLATE",
    "AGENT_PLAN_TEMPLATE",
    "AGENT_DECISION_TEMPLATE",
    "SMOL_AGENT_TASK_TEMPLATE",
    "COLLABORATION_TEMPLATE",
]
