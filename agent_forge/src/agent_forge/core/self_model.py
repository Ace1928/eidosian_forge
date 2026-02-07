from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from eidosian_core import eidosian

from . import state as state_core
from . import workspace

__all__ = ["snapshot", "emit_snapshot"]


def _memory_stats(memory_dir: Path) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    try:
        from memory_forge.core.introspection import MemoryIntrospector
    except ImportError:
        return None, []

    introspector = MemoryIntrospector(memory_dir)
    stats = introspector.get_stats()
    insights = introspector.analyze_patterns()
    stats_dict = {
        "total_memories": stats.total_memories,
        "by_tier": stats.by_tier,
        "by_namespace": stats.by_namespace,
        "by_type": stats.by_type,
        "top_tags": stats.top_tags,
        "avg_importance": stats.avg_importance,
        "avg_access_count": stats.avg_access_count,
        "oldest_memory": stats.oldest_memory.isoformat() if stats.oldest_memory else None,
        "newest_memory": stats.newest_memory.isoformat() if stats.newest_memory else None,
    }
    insight_dicts = [
        {
            "type": i.insight_type,
            "description": i.description,
            "confidence": i.confidence,
            "evidence": i.evidence,
            "metadata": i.metadata,
            "timestamp": i.timestamp.isoformat(),
        }
        for i in insights
    ]
    return stats_dict, insight_dicts


@eidosian()
def snapshot(
    *,
    state_dir: str | Path = "state",
    memory_dir: str | Path = "/home/lloyd/eidosian_forge/data/memory",
    last_events: int = 5,
    window_seconds: float = 1.0,
    min_sources: int = 3,
) -> Dict[str, Any]:
    memory_path = Path(memory_dir)
    memory_stats, memory_insights = _memory_stats(memory_path)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "state": state_core.snapshot(str(state_dir), last=last_events),
        "workspace": workspace.summary(
            state_dir,
            window_seconds=window_seconds,
            min_sources=min_sources,
        ),
        "memory": memory_stats,
        "memory_insights": memory_insights,
    }


@eidosian()
def emit_snapshot(
    *,
    state_dir: str | Path = "state",
    memory_dir: str | Path = "/home/lloyd/eidosian_forge/data/memory",
    last_events: int = 5,
    window_seconds: float = 1.0,
    min_sources: int = 3,
) -> Dict[str, Any]:
    snap = snapshot(
        state_dir=state_dir,
        memory_dir=memory_dir,
        last_events=last_events,
        window_seconds=window_seconds,
        min_sources=min_sources,
    )
    workspace.broadcast(
        state_dir,
        source="self_model",
        payload=snap,
        tags=["self_model", "snapshot"],
    )
    return snap
