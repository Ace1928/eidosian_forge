from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from eidosian_core import eidosian

from . import events, workspace
from . import state as state_core

__all__ = ["snapshot", "emit_snapshot"]


FORGE_ROOT = Path(
    os.environ.get(
        "EIDOS_FORGE_DIR",
        str(Path(__file__).resolve().parents[4]),
    )
).resolve()


def _memory_stats(memory_dir: Path) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    try:
        from memory_forge.core.introspection import MemoryIntrospector
    except Exception:
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


def _latest_event_data(items: List[Dict[str, Any]], etype: str) -> Optional[Dict[str, Any]]:
    for evt in reversed(items):
        if str(evt.get("type") or "") == etype:
            data = evt.get("data")
            if isinstance(data, dict):
                return data
    return None


def _latest_metric(items: List[Dict[str, Any]], key: str) -> Optional[float]:
    for evt in reversed(items):
        if str(evt.get("type") or "") != "metrics.sample":
            continue
        data = evt.get("data")
        if not isinstance(data, dict):
            continue
        if str(data.get("key") or "") != key:
            continue
        try:
            return float(data.get("value"))
        except (TypeError, ValueError):
            return None
    return None


def _recent_winners(items: List[Dict[str, Any]], *, limit: int = 5) -> List[Dict[str, Any]]:
    winners: List[Dict[str, Any]] = []
    for evt in items:
        if str(evt.get("type") or "") != "workspace.broadcast":
            continue
        data = evt.get("data")
        if not isinstance(data, dict):
            continue
        payload = data.get("payload")
        if not isinstance(payload, dict):
            continue
        if str(payload.get("kind") or "") != "GW_WINNER":
            continue
        content = payload.get("content") if isinstance(payload.get("content"), dict) else {}
        winners.append(
            {
                "ts": evt.get("ts"),
                "candidate_id": content.get("candidate_id"),
                "source_event_type": content.get("source_event_type"),
                "source_module": content.get("source_module"),
                "score": content.get("score"),
            }
        )
    return winners[-limit:]


def _consciousness_snapshot(state_dir: Path, *, last_events: int = 300) -> Dict[str, Any]:
    items = events.iter_events(state_dir, limit=last_events)
    latest_agency = _latest_event_data(items, "self.agency_estimate")
    latest_boundary = _latest_event_data(items, "self.boundary_estimate")
    latest_meta = _latest_event_data(items, "meta.state_estimate")
    latest_report = _latest_event_data(items, "report.self_report")
    latest_memory_bridge = _latest_event_data(items, "memory_bridge.status")
    latest_knowledge_bridge = _latest_event_data(items, "knowledge_bridge.status")
    latest_memory_recall = _latest_event_data(items, "mem.recall")
    latest_knowledge_context = _latest_event_data(items, "knowledge.context")

    return {
        "agency": {
            "confidence": (latest_agency or {}).get("agency_confidence"),
            "action_id": (latest_agency or {}).get("action_id"),
            "predicted": (latest_agency or {}).get("predicted"),
            "observed": (latest_agency or {}).get("observed"),
            "metric": _latest_metric(items, "consciousness.agency"),
        },
        "boundary": {
            "stability": (latest_boundary or {}).get("boundary_stability"),
            "control_graph": (latest_boundary or {}).get("control_graph"),
            "metric": _latest_metric(items, "consciousness.boundary_stability"),
        },
        "meta": latest_meta or {},
        "latest_report": latest_report or {},
        "integration": {
            "memory_bridge": latest_memory_bridge or {},
            "knowledge_bridge": latest_knowledge_bridge or {},
            "latest_memory_recall": latest_memory_recall or {},
            "latest_knowledge_context": latest_knowledge_context or {},
        },
        "recent_winners": _recent_winners(items, limit=5),
    }


@eidosian()
def snapshot(
    *,
    state_dir: str | Path = "state",
    memory_dir: str | Path = os.environ.get("EIDOS_MEMORY_DIR", str(FORGE_ROOT / "data" / "memory")),
    last_events: int = 5,
    window_seconds: float = 1.0,
    min_sources: int = 3,
) -> Dict[str, Any]:
    state_path = Path(state_dir)
    memory_path = Path(memory_dir)
    memory_stats, memory_insights = _memory_stats(memory_path)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "state": state_core.snapshot(str(state_path), last=last_events),
        "workspace": workspace.summary(
            state_path,
            window_seconds=window_seconds,
            min_sources=min_sources,
        ),
        "memory": memory_stats,
        "memory_insights": memory_insights,
        "consciousness": _consciousness_snapshot(state_path, last_events=max(300, last_events * 20)),
    }


@eidosian()
def emit_snapshot(
    *,
    state_dir: str | Path = "state",
    memory_dir: str | Path = os.environ.get("EIDOS_MEMORY_DIR", str(FORGE_ROOT / "data" / "memory")),
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
