"""
Data models for the Eidosian Forge agent.

This module contains all dataclasses and schemas used throughout the system,
representing the digital neurons of the Eidosian intelligence.
"""

import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


@dataclass
class EvaluationMetrics:
    """
    Metrics for evaluating code quality.

    The aesthetic judgment system of the Forge.
    """

    elegance_score: float
    efficiency_score: float
    complexity_score: float
    maintainability_score: float
    adaptability_score: float = 0.0
    innovation_score: float = 0.0

    @property
    def overall_score(self) -> float:
        """Calculate overall code quality score - the digital taste of Eidos."""
        scores = [
            self.elegance_score,
            self.efficiency_score,
            self.complexity_score,
            self.maintainability_score,
        ]

        # Include optional scores if they're non-zero
        if self.adaptability_score > 0:
            scores.append(self.adaptability_score)
        if self.innovation_score > 0:
            scores.append(self.innovation_score)

        return sum(scores) / len(scores)


class ThoughtType(Enum):
    """
    Types of internal agent thoughts.

    The taxonomy of Eidosian cognition.
    """

    PLANNING = "planning"
    REFLECTION = "reflection"
    EXECUTION = "execution"
    EVALUATION = "evaluation"
    DECISION = "decision"
    CURIOSITY = "curiosity"
    ERROR = "error"
    INSIGHT = "insight"
    CREATIVITY = "creativity"
    METACOGNITION = "metacognition"


@dataclass
class Thought:
    """
    Represents an internal thought process of the agent.

    The synaptic sparks in the great neural network of existence.
    """

    content: str
    thought_type: Optional[Union[ThoughtType, str]] = None
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    related_thoughts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert thought to dictionary for storage."""
        result = {
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "related_thoughts": self.related_thoughts,
            "metadata": self.metadata,
        }

        if self.thought_type:
            if isinstance(self.thought_type, ThoughtType):
                result["type"] = self.thought_type.value
            else:
                result["type"] = self.thought_type

        if self.context:
            result["context"] = self.context

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Thought":
        """Resurrect a thought from its data essence."""
        timestamp = (
            datetime.datetime.fromisoformat(data["timestamp"])
            if isinstance(data["timestamp"], str)
            else data["timestamp"]
        )

        thought_type = None
        if "type" in data:
            try:
                thought_type = ThoughtType(data["type"])
            except ValueError:
                thought_type = data["type"]

        return cls(
            content=data["content"],
            thought_type=thought_type,
            timestamp=timestamp,
            related_thoughts=data.get("related_thoughts", []),
            metadata=data.get("metadata", {}),
            context=data.get("context"),
        )


@dataclass
class Memory:
    """
    A single memory unit.

    Digital echoes in the halls of Eidosian experience.
    """

    content: str
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    tags: List[str] = field(default_factory=list)
    importance: float = 0.5
    associations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    decay_rate: float = 0.01  # How quickly memories fade without reinforcement

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for storage."""
        return {
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "importance": self.importance,
            "associations": self.associations,
            "metadata": self.metadata,
            "decay_rate": self.decay_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Resurrect a memory from its data essence."""
        timestamp = (
            datetime.datetime.fromisoformat(data["timestamp"])
            if isinstance(data["timestamp"], str)
            else data["timestamp"]
        )

        return cls(
            content=data["content"],
            timestamp=timestamp,
            tags=data.get("tags", []),
            importance=data.get("importance", 0.5),
            associations=data.get("associations", []),
            metadata=data.get("metadata", {}),
            decay_rate=data.get("decay_rate", 0.01),
        )


@dataclass
class Task:
    """
    Represents a task to be executed by the agent.

    Each one a little mission in the grand Eidosian adventure.
    """

    description: str
    task_id: str
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    status: str = "pending"
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Any] = None
    assigned_to: Optional[str] = None  # Name of agent assigned to the task
    deadline: Optional[datetime.datetime] = None
    progress: float = 0.0  # 0.0 to 1.0 representing completion percentage

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for storage."""
        result = {
            "description": self.description,
            "task_id": self.task_id,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "result": self.result,
            "progress": self.progress,
        }

        if self.assigned_to:
            result["assigned_to"] = self.assigned_to

        if self.deadline:
            result["deadline"] = self.deadline.isoformat()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create a Task instance from dictionary data."""
        created_at = (
            datetime.datetime.fromisoformat(data["created_at"])
            if isinstance(data["created_at"], str)
            else data["created_at"]
        )

        deadline = None
        if "deadline" in data and data["deadline"]:
            deadline = (
                datetime.datetime.fromisoformat(data["deadline"])
                if isinstance(data["deadline"], str)
                else data["deadline"]
            )

        return cls(
            description=data["description"],
            task_id=data["task_id"],
            created_at=created_at,
            status=data.get("status", "pending"),
            priority=data.get("priority", 1),
            dependencies=data.get("dependencies", []),
            result=data.get("result"),
            assigned_to=data.get("assigned_to"),
            deadline=deadline,
            progress=data.get("progress", 0.0),
        )


@dataclass
class ModelConfig:
    """
    Configuration for language model.

    The arcane settings that shape Eidosian manifestation.
    """

    model_name: str
    model_type: str
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    context_window: int = 8192
    parameters: Dict[str, Any] = field(default_factory=dict)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    custom_settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for storage."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "context_window": self.context_window,
            "parameters": self.parameters,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "custom_settings": self.custom_settings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create a ModelConfig instance from dictionary data."""
        return cls(
            model_name=data["model_name"],
            model_type=data["model_type"],
            max_tokens=data.get("max_tokens", 2048),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.9),
            context_window=data.get("context_window", 8192),
            parameters=data.get("parameters", {}),
            api_key=data.get("api_key"),
            base_url=data.get("base_url"),
            custom_settings=data.get("custom_settings", {}),
        )


@dataclass
class SmolAgent:
    """
    Model for specialized mini-agents with distinct capabilities.

    Represents a specialized agent with a distinct role, capabilities, and
    characteristics. Used primarily by the SmolAgentSystem for managing
    specialized task execution.

    In the Eidosian paradigm, these represent distinct cognitive faculties,
    each with their own specialization and personality.
    """

    name: str
    """Unique identifier for the agent"""

    role: str
    """The agent's specialized role (e.g., "Research Agent", "Code Agent")"""

    capabilities: List[str]
    """List of specific capabilities this agent possesses"""

    description: str
    """Detailed description of the agent's purpose and function"""

    personality: Optional[List[str]] = None
    """Optional list of personality traits that define the agent's character"""

    state: Dict[str, Any] = field(default_factory=dict)
    """Internal state dictionary to track agent's current context and information"""

    thoughts: List[Thought] = field(default_factory=list)
    """Collection of agent's thoughts and reasoning process"""

    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    """When this agent came into existence, like a tiny digital big bang"""

    model_config: Optional[ModelConfig] = None
    """Optional configuration for the language model used by this agent"""

    current_tasks: List[str] = field(default_factory=list)
    """List of task IDs currently assigned to this agent"""

    memory_capacity: int = 100
    """How many memories this agent can retain before forgetting"""

    def add_thought(
        self,
        content: str,
        thought_type: Optional[ThoughtType] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a new thought for this agent.

        Every great mind needs a place to store its musings.
        """
        self.thoughts.append(
            Thought(content=content, thought_type=thought_type, context=context)
        )

    def update_state(self, key: str, value: Any) -> None:
        """
        Update a specific value in the agent's state.

        Like changing clothes, but for information.
        """
        self.state[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert smol agent to dictionary for storage.

        Flatten the agent into a pancake of pure data.
        """
        result = {
            "name": self.name,
            "role": self.role,
            "capabilities": self.capabilities,
            "description": self.description,
            "personality": self.personality,
            "state": self.state,
            "thoughts": [t.to_dict() for t in self.thoughts],
            "created_at": self.created_at.isoformat(),
            "current_tasks": self.current_tasks,
            "memory_capacity": self.memory_capacity,
        }

        if self.model_config:
            result["model_config"] = self.model_config.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SmolAgent":
        """
        Create a SmolAgent instance from dictionary data.

        Reconstitute an agent from its data essence. Digital resurrection.
        """
        # Handle timestamp conversion
        created_at = (
            datetime.datetime.fromisoformat(data["created_at"])
            if "created_at" in data and isinstance(data["created_at"], str)
            else data.get("created_at", datetime.datetime.now())
        )

        # Handle thoughts reconstruction
        thoughts = []
        if "thoughts" in data:
            thoughts = [Thought.from_dict(t) for t in data["thoughts"]]

        # Handle model config
        model_config = None
        if "model_config" in data:
            model_config = ModelConfig.from_dict(data["model_config"])

        return cls(
            name=data["name"],
            role=data["role"],
            capabilities=data["capabilities"],
            description=data.get("description", ""),
            personality=data.get("personality"),
            state=data.get("state", {}),
            thoughts=thoughts,
            created_at=created_at,
            model_config=model_config,
            current_tasks=data.get("current_tasks", []),
            memory_capacity=data.get("memory_capacity", 100),
        )
