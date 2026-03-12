from eidosian_core import eidosian

"""
Memory system for Eidosian Forge.

Provides persistent, versioned memory storage with retrieval capabilities.
"""

import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from agent_forge.models import Memory, Task, Thought, ThoughtType

# Import GitMemoryManager only if available
try:
    from agent_forge.utils import GitMemoryManager
except ImportError:
    GitMemoryManager = None

logger = logging.getLogger(__name__)


def _forge_root() -> Path:
    raw = os.environ.get("EIDOS_FORGE_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


def _ensure_tiered_import_path() -> None:
    root = _forge_root()
    for path in (
        root / "lib",
        root / "memory_forge" / "src",
        root / "knowledge_forge" / "src",
        root / "eidos_mcp" / "src",
        root / "ollama_forge" / "src",
        root,
    ):
        text = str(path)
        if path.exists() and text not in sys.path:
            sys.path.insert(0, text)


class MemorySystem:
    """
    Memory system for the Eidosian AI.

    Provides storage and retrieval of memories, thoughts, and tasks with
    versioning through Git integration.
    """

    def __init__(
        self,
        memory_path: str,
        git_enabled: bool = True,
        auto_commit: bool = True,
        commit_interval_minutes: int = 30,
        tiered_memory_system: Any = None,
    ):
        """
        Initialize memory system.

        Args:
            memory_path: Base path for memory storage
            git_enabled: Whether to use Git for versioning
            auto_commit: Whether to automatically commit changes
            commit_interval_minutes: Interval between auto-commits in minutes
        """
        self.memory_path = Path(memory_path)
        self.git_enabled = git_enabled and GitMemoryManager is not None
        self._tiered_memory_system = tiered_memory_system

        if self.git_enabled:
            self.git = GitMemoryManager(
                repo_path=memory_path,
                auto_commit=auto_commit,
                commit_interval_minutes=commit_interval_minutes,
            )
        else:
            os.makedirs(memory_path, exist_ok=True)
            # Create directory structure even without Git
            memory_dirs = [
                "thoughts",
                "reflections",
                "tasks",
                "knowledge",
                "changes",
                "memories",
            ]
            for d in memory_dirs:
                os.makedirs(self.memory_path / d, exist_ok=True)
            logger.info(f"Git integration disabled or unavailable, using directory-based storage at {memory_path}")

    def _tiered_memory_dir(self) -> Path:
        override = os.environ.get("EIDOS_MEMORY_DIR")
        if override:
            return Path(override).expanduser().resolve()
        root = _forge_root()
        preferred = root / "data" / "tiered_memory"
        legacy = root / "data" / "memory"
        if preferred.exists() or not legacy.exists():
            return preferred
        if legacy.exists():
            return legacy
        return (self.memory_path / "tiered").resolve()

    def _load_tiered_memory_system(self) -> Any:
        if self._tiered_memory_system is not None:
            return self._tiered_memory_system
        try:
            _ensure_tiered_import_path()
            from eidosian_vector import build_default_embedder  # type: ignore

            from memory_forge import TieredMemorySystem  # type: ignore

            self._tiered_memory_system = TieredMemorySystem(
                persistence_dir=self._tiered_memory_dir(),
                embedder=build_default_embedder(),
            )
        except Exception:
            self._tiered_memory_system = None
        return self._tiered_memory_system

    def _legacy_search_thoughts(self, query: str, max_results: int) -> List[Thought]:
        results = []
        thought_ids = []
        if self.git_enabled and hasattr(self, "git"):
            thought_ids = self.git.list_memories("thoughts")
        else:
            thought_dir = self.memory_path / "thoughts"
            if thought_dir.exists():
                thought_ids = [f.stem for f in thought_dir.glob("*.json") if f.is_file() and not f.name.startswith(".")]

        for thought_id in thought_ids:
            thought = self.get_thought(thought_id)
            if thought and query in thought.content.lower():
                results.append(thought)
                if len(results) >= max_results:
                    break
        return results

    @staticmethod
    def _thought_from_tiered(item: Any) -> Thought:
        metadata = dict(getattr(item, "metadata", {}) or {})
        thought_type = metadata.get("thought_type") or metadata.get("type")
        return Thought(
            content=str(getattr(item, "content", "")),
            thought_type=thought_type,
            timestamp=getattr(item, "created_at", datetime.now()),
            related_thoughts=sorted(list(getattr(item, "linked_memories", set()) or [])),
            metadata={
                **metadata,
                "source": "tiered_memory",
                "memory_id": str(getattr(item, "id", "")),
                "tier": getattr(getattr(item, "tier", None), "value", getattr(item, "tier", "")),
                "namespace": getattr(getattr(item, "namespace", None), "value", getattr(item, "namespace", "")),
                "importance": getattr(item, "importance", 0.5),
                "tags": sorted(list(getattr(item, "tags", set()) or [])),
            },
        )

    @staticmethod
    def _memory_from_tiered(item: Any) -> Memory:
        metadata = dict(getattr(item, "metadata", {}) or {})
        tags = sorted(list(getattr(item, "tags", set()) or []))
        associations = sorted(list(getattr(item, "linked_memories", set()) or []))
        return Memory(
            content=str(getattr(item, "content", "")),
            timestamp=getattr(item, "created_at", datetime.now()),
            tags=tags,
            importance=float(getattr(item, "importance", 0.5) or 0.5),
            associations=associations,
            metadata={
                **metadata,
                "source": "tiered_memory",
                "memory_id": str(getattr(item, "id", "")),
                "tier": getattr(getattr(item, "tier", None), "value", getattr(item, "tier", "")),
                "namespace": getattr(getattr(item, "namespace", None), "value", getattr(item, "namespace", "")),
            },
        )

    @eidosian()
    def save_thought(self, thought: Thought) -> str:
        """
        Save a thought to memory.

        Args:
            thought: Thought object to save

        Returns:
            ID of saved thought
        """
        thought_id = f"{thought.timestamp.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

        if self.git_enabled and hasattr(self, "git"):
            self.git.save_memory("thoughts", thought_id, thought.to_dict())
        else:
            # Save without Git
            thought_dir = self.memory_path / "thoughts"
            with open(thought_dir / f"{thought_id}.json", "w") as f:
                json.dump(thought.to_dict(), f, indent=2)

        logger.debug(f"Saved thought: {thought_id}")
        return thought_id

    # Add an alias for backward compatibility
    add_thought = save_thought

    @eidosian()
    def save_memory(self, memory: Memory) -> str:
        """
        Save a memory.

        Args:
            memory: Memory object to save

        Returns:
            ID of saved memory
        """
        memory_id = f"{memory.timestamp.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

        if self.git_enabled and hasattr(self, "git"):
            self.git.save_memory("memories", memory_id, memory.to_dict())
        else:
            # Save without Git
            memory_dir = self.memory_path / "memories"
            os.makedirs(memory_dir, exist_ok=True)
            with open(memory_dir / f"{memory_id}.json", "w") as f:
                json.dump(memory.to_dict(), f, indent=2)

        logger.debug(f"Saved memory: {memory_id}")
        return memory_id

    @eidosian()
    def save_task(self, task: Task) -> str:
        """
        Save a task.

        Args:
            task: Task object to save

        Returns:
            ID of saved task
        """
        if not task.task_id:
            task.task_id = f"task_{uuid.uuid4().hex[:8]}"

        if self.git_enabled and hasattr(self, "git"):
            self.git.save_memory("tasks", task.task_id, task.to_dict())
        else:
            # Save without Git
            task_dir = self.memory_path / "tasks"
            with open(task_dir / f"{task.task_id}.json", "w") as f:
                json.dump(task.to_dict(), f, indent=2)

        logger.debug(f"Saved task: {task.task_id}")
        return task.task_id

    @eidosian()
    def get_thought(self, thought_id: str) -> Optional[Thought]:
        """
        Retrieve a thought by ID.

        Args:
            thought_id: ID of thought to retrieve

        Returns:
            Thought object if found, None otherwise
        """
        data = None
        if self.git_enabled and hasattr(self, "git"):
            data = self.git.load_memory("thoughts", thought_id)
        else:
            thought_path = self.memory_path / "thoughts" / f"{thought_id}.json"
            if not thought_path.exists():
                return None

            with open(thought_path, "r") as f:
                data = json.load(f)

        if data:
            # Convert back to Thought object
            thought_type_str = data.get("type")
            # Handle the case where thought_type might be None
            thought_type = ThoughtType(thought_type_str) if thought_type_str else ThoughtType.UNKNOWN

            return Thought(
                content=data["content"],
                thought_type=thought_type,
                timestamp=datetime.fromisoformat(data["timestamp"]),
                related_thoughts=data.get("related_thoughts", []),
                metadata=data.get("metadata", {}),
            )

        return None

    @eidosian()
    def get_recent_thoughts(self, n: int = 10, thought_type: Optional[str] = None) -> List[Thought]:
        """
        Get most recent thoughts.

        Args:
            n: Number of thoughts to retrieve
            thought_type: Optional filter by thought type

        Returns:
            List of Thought objects
        """
        thought_ids = []
        if self.git_enabled and hasattr(self, "git"):
            thought_ids = self.git.list_memories("thoughts")
        else:
            thought_dir = self.memory_path / "thoughts"
            if thought_dir.exists():
                thought_ids = [f.stem for f in thought_dir.glob("*.json") if f.is_file() and not f.name.startswith(".")]

        # Sort by timestamp (which is encoded in the ID)
        thought_ids.sort(reverse=True)

        # Limit to n thoughts
        thought_ids = thought_ids[:n]

        thoughts = []
        for thought_id in thought_ids:
            thought = self.get_thought(thought_id)
            if thought and (
                not thought_type
                or (hasattr(thought.thought_type, "value") and thought.thought_type.value == thought_type)
            ):
                thoughts.append(thought)

        return thoughts

    @eidosian()
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Retrieve a task by ID.

        Args:
            task_id: ID of task to retrieve

        Returns:
            Task object if found, None otherwise
        """
        data = None
        if self.git_enabled and hasattr(self, "git"):
            data = self.git.load_memory("tasks", task_id)
        else:
            task_path = self.memory_path / "tasks" / f"{task_id}.json"
            if not task_path.exists():
                return None

            with open(task_path, "r") as f:
                data = json.load(f)

        if data:
            # Convert back to Task object
            return Task(
                description=data["description"],
                task_id=data["task_id"],
                created_at=datetime.fromisoformat(data["created_at"]),
                status=data.get("status", "pending"),
                priority=data.get("priority", 1),
                dependencies=data.get("dependencies", []),
                result=data.get("result"),
            )

        return None

    @eidosian()
    def search_thoughts(self, query: str, max_results: int = 10) -> List[Thought]:
        """
        Search for thoughts containing the query string.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of matching Thought objects
        """
        query_text = str(query or "").strip()
        if not query_text:
            return []

        results: List[Thought] = []
        seen: set[str] = set()

        tiered = self._load_tiered_memory_system()
        if tiered is not None and hasattr(tiered, "recall"):
            try:
                for item in tiered.recall(query_text, limit=max_results):
                    thought = self._thought_from_tiered(item)
                    signature = thought.metadata.get("memory_id") or thought.content
                    if signature in seen:
                        continue
                    seen.add(str(signature))
                    results.append(thought)
                    if len(results) >= max_results:
                        return results
            except Exception:
                logger.debug("Tiered thought search failed", exc_info=True)

        for thought in self._legacy_search_thoughts(query_text.lower(), max_results=max_results):
            signature = thought.content
            if signature in seen:
                continue
            seen.add(signature)
            results.append(thought)
            if len(results) >= max_results:
                break

        return results

    # Alias for search_memories method expected by some components
    search_memories = search_thoughts

    @eidosian()
    def get_memories(self, query: str, max_results: int = 10) -> List[Memory]:
        """
        Retrieve memories that match a query.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of matching Memory objects
        """
        query_text = str(query or "").strip()
        if not query_text:
            return []
        tiered = self._load_tiered_memory_system()
        if tiered is None or not hasattr(tiered, "recall"):
            return []
        try:
            return [self._memory_from_tiered(item) for item in tiered.recall(query_text, limit=max_results)]
        except Exception:
            logger.debug("Tiered memory retrieval failed", exc_info=True)
            return []

    @eidosian()
    def commit_changes(self, message: str) -> bool:
        """
        Manually commit changes to Git repository.

        Args:
            message: Commit message

        Returns:
            True if successful, False otherwise
        """
        if not self.git_enabled or not hasattr(self, "git"):
            logger.warning("Git is not enabled, cannot commit changes")
            return False

        return self.git.commit(message)

    @eidosian()
    def close(self) -> None:
        """Clean up resources."""
        if self.git_enabled and hasattr(self, "git"):
            self.git.close()
        logger.info("Memory system closed")
