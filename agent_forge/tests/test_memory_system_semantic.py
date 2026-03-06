from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from agent_forge.core.memory import MemorySystem


@dataclass
class _FakeTieredItem:
    content: str
    id: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)
    linked_memories: set[str] = field(default_factory=set)
    importance: float = 0.8
    tier: str = "working"
    namespace: str = "task"


class _FakeTieredMemory:
    def recall(self, query: str, limit: int = 10):
        _ = query
        return [
            _FakeTieredItem(
                content="Vector memory about autonomy",
                id="mem-1",
                metadata={"thought_type": "planning"},
                tags={"autonomy", "vector"},
            ),
            _FakeTieredItem(
                content="Knowledge memory about graphs",
                id="mem-2",
                metadata={"category": "knowledge"},
                tags={"graph"},
            ),
        ][:limit]


def test_memory_system_search_thoughts_uses_tiered_memory(tmp_path) -> None:
    memory = MemorySystem(str(tmp_path / "memory_repo"), git_enabled=False, tiered_memory_system=_FakeTieredMemory())

    results = memory.search_thoughts("autonomy vector", max_results=5)

    assert len(results) == 2
    assert results[0].content == "Vector memory about autonomy"
    assert results[0].metadata["source"] == "tiered_memory"
    assert results[0].thought_type == "planning"


def test_memory_system_get_memories_uses_tiered_memory(tmp_path) -> None:
    memory = MemorySystem(str(tmp_path / "memory_repo"), git_enabled=False, tiered_memory_system=_FakeTieredMemory())

    results = memory.get_memories("graphs", max_results=5)

    assert len(results) == 2
    assert results[1].content == "Knowledge memory about graphs"
    assert results[1].metadata["source"] == "tiered_memory"
    assert "graph" in results[1].tags
