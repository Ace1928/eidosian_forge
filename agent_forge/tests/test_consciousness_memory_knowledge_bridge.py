from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent_forge.consciousness.kernel import ConsciousnessKernel
from agent_forge.consciousness.modules.knowledge_bridge import KnowledgeBridgeModule
from agent_forge.consciousness.modules.memory_bridge import MemoryBridgeModule
from agent_forge.core import events, self_model, workspace


class _FakeMemoryItem:
    def __init__(
        self,
        item_id: str,
        content: str,
        *,
        tier: str = "working",
        namespace: str = "task",
        importance: float = 0.8,
        access_count: int = 0,
        tags: set[str] | None = None,
    ) -> None:
        self.id = item_id
        self.content = content
        self.tier = SimpleNamespace(value=tier)
        self.namespace = SimpleNamespace(value=namespace)
        self.importance = importance
        self.access_count = access_count
        self.tags = tags or set()


class _FakeMemorySystem:
    def recall(self, query: str, limit: int = 4) -> list[_FakeMemoryItem]:
        _ = query
        return [
            _FakeMemoryItem(
                "mem-1",
                "agent forge workspace winner memory",
                importance=0.9,
                tags={"agent", "workspace"},
            ),
            _FakeMemoryItem(
                "mem-2",
                "knowledge integration baseline",
                importance=0.7,
                tags={"knowledge"},
            ),
        ][:limit]


class _FakeIntrospector:
    def get_stats(self) -> Any:
        return SimpleNamespace(
            total_memories=2,
            by_tier={"working": 2},
            by_namespace={"task": 2},
            by_type={"episodic": 2},
            avg_importance=0.8,
            avg_access_count=0.2,
            top_tags=[("agent", 1), ("knowledge", 1)],
        )

    def analyze_patterns(self) -> list[dict[str, Any]]:
        return [
            {
                "insight_type": "pattern",
                "description": "Working memories dominate active context",
                "confidence": 0.82,
                "evidence": ["mem-1"],
            }
        ]


class _FakeBridge:
    def get_memory_knowledge_context(
        self,
        query: str,
        max_results: int = 6,
    ) -> dict[str, Any]:
        _ = query
        memory_context = [
            {
                "id": "mem-1",
                "content": "agent forge workspace winner memory",
                "score": 0.88,
                "tier": "working",
                "namespace": "task",
            }
        ]
        knowledge_context = [
            {
                "id": "kb-1",
                "content": "Global workspace model uses winner selection and ignition",
                "score": 0.93,
                "tags": ["gnw", "workspace"],
            }
        ]
        merged = (memory_context + knowledge_context)[:max_results]
        return {
            "query": query,
            "total_results": len(merged),
            "memory_context": memory_context[:max_results],
            "knowledge_context": knowledge_context[:max_results],
        }

    def stats(self) -> dict[str, Any]:
        return {
            "memory_count": 3,
            "knowledge_count": 2,
            "memory_to_knowledge_links": 1,
            "knowledge_to_memory_links": 1,
        }


def _winner_payload(candidate_id: str = "cand-1") -> dict[str, Any]:
    return {
        "kind": "GW_WINNER",
        "ts": "2026-01-01T00:00:00Z",
        "id": "winner-1",
        "source_module": "workspace_competition",
        "confidence": 0.9,
        "salience": 0.85,
        "content": {
            "candidate_id": candidate_id,
            "source_event_type": "sense.percept",
            "source_module": "sense",
            "score": 0.91,
        },
        "links": {"corr_id": "c1", "parent_id": "p1", "memory_ids": []},
    }


def test_memory_bridge_emits_recall_and_broadcast(tmp_path: Path) -> None:
    base = tmp_path / "state"
    workspace.broadcast(base, "workspace_competition", _winner_payload())

    kernel = ConsciousnessKernel(
        base,
        modules=[
            MemoryBridgeModule(
                memory_system=_FakeMemorySystem(),
                introspector=_FakeIntrospector(),
            )
        ],
        config={"memory_bridge_stats_period_beats": 1},
        seed=11,
    )
    result = kernel.tick()

    assert result.errors == []
    all_events = events.iter_events(base, limit=None)
    assert any(evt.get("type") == "mem.recall" for evt in all_events)
    assert any(evt.get("type") == "memory_bridge.status" for evt in all_events)
    assert any(
        evt.get("type") == "workspace.broadcast"
        and (((evt.get("data") or {}).get("payload") or {}).get("kind") == "MEMORY")
        for evt in all_events
    )


def test_knowledge_bridge_emits_context_recall_and_broadcast(tmp_path: Path) -> None:
    base = tmp_path / "state"
    events.append(
        base,
        "mem.recall",
        {
            "query": "workspace ignition",
            "content": "agent forge workspace winner memory",
            "namespace": "task",
            "tier": "working",
        },
    )

    kernel = ConsciousnessKernel(
        base,
        modules=[KnowledgeBridgeModule(bridge=_FakeBridge())],
        seed=13,
    )
    result = kernel.tick()

    assert result.errors == []
    all_events = events.iter_events(base, limit=None)
    assert any(evt.get("type") == "knowledge.context" for evt in all_events)
    assert any(evt.get("type") == "knowledge.recall" for evt in all_events)
    assert any(evt.get("type") == "knowledge_bridge.status" for evt in all_events)
    assert any(
        evt.get("type") == "workspace.broadcast"
        and (((evt.get("data") or {}).get("payload") or {}).get("kind") == "KNOWLEDGE")
        for evt in all_events
    )


def test_kernel_default_module_order_includes_bridges(tmp_path: Path) -> None:
    base = tmp_path / "state"
    kernel = ConsciousnessKernel(base, seed=5)
    names = [module.name for module in kernel.modules]

    assert "memory_bridge" in names
    assert "knowledge_bridge" in names
    assert names.index("memory_bridge") < names.index("attention")
    assert names.index("knowledge_bridge") < names.index("attention")


def test_self_model_snapshot_includes_bridge_integration_fields(tmp_path: Path) -> None:
    base = tmp_path / "state"
    events.append(
        base,
        "memory_bridge.status",
        {
            "available": True,
            "query": "workspace",
            "recall_count": 2,
            "last_error": "",
        },
    )
    events.append(
        base,
        "knowledge_bridge.status",
        {
            "available": True,
            "query": "workspace",
            "total_hits": 3,
            "last_error": "",
        },
    )
    events.append(
        base,
        "knowledge.context",
        {
            "query": "workspace",
            "total_hits": 3,
            "memory_hits": 1,
            "knowledge_hits": 2,
        },
    )

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    snap = self_model.snapshot(state_dir=base, memory_dir=memory_dir, last_events=30)
    integration = (snap.get("consciousness") or {}).get("integration") or {}

    assert (integration.get("memory_bridge") or {}).get("available") is True
    assert (integration.get("knowledge_bridge") or {}).get("available") is True
    assert (integration.get("latest_knowledge_context") or {}).get("total_hits") == 3
