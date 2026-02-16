from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence

from agent_forge.core import db as DB
from agent_forge.core import events as bus
from agent_forge.core import workspace


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clamp01(value: Any, *, default: float = 0.5) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


@dataclass(frozen=True)
class WorkspacePayload:
    kind: str
    source_module: str
    content: Dict[str, Any]
    confidence: float = 0.5
    salience: float = 0.5
    ts: str = field(default_factory=_now_iso)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    links: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "ts": self.ts,
            "id": self.id,
            "source_module": self.source_module,
            "confidence": clamp01(self.confidence),
            "salience": clamp01(self.salience),
            "content": dict(self.content),
            "links": dict(self.links),
        }


def normalize_workspace_payload(payload: Mapping[str, Any], *, fallback_kind: str, source_module: str) -> Dict[str, Any]:
    links = payload.get("links") if isinstance(payload.get("links"), Mapping) else {}
    normalized = {
        "kind": str(payload.get("kind") or fallback_kind),
        "ts": str(payload.get("ts") or _now_iso()),
        "id": str(payload.get("id") or uuid.uuid4().hex),
        "source_module": str(payload.get("source_module") or source_module),
        "confidence": clamp01(payload.get("confidence"), default=0.5),
        "salience": clamp01(payload.get("salience"), default=0.5),
        "content": dict(payload.get("content") or {}),
        "links": {
            "corr_id": str(links.get("corr_id") or ""),
            "parent_id": str(links.get("parent_id") or ""),
            "memory_ids": list(links.get("memory_ids") or []),
        },
    }
    return normalized


class Module(Protocol):
    name: str

    def tick(self, ctx: "TickContext") -> None:
        ...


@dataclass
class TickContext:
    state_dir: Path
    config: Mapping[str, Any]
    recent_events: Sequence[Dict[str, Any]]
    recent_broadcasts: Sequence[Dict[str, Any]]
    rng: Any
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    emitted_events: list[Dict[str, Any]] = field(default_factory=list)

    def emit_event(
        self,
        etype: str,
        data: Optional[Mapping[str, Any]] = None,
        *,
        tags: Optional[Sequence[str]] = None,
        corr_id: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        evt = bus.append(
            self.state_dir,
            etype,
            dict(data or {}),
            tags=list(tags or []),
            corr_id=corr_id,
            parent_id=parent_id,
        )
        self.emitted_events.append(evt)
        return evt

    def broadcast(
        self,
        source: str,
        payload: Mapping[str, Any],
        *,
        channel: str = "global",
        tags: Optional[Sequence[str]] = None,
        corr_id: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        evt = workspace.broadcast(
            self.state_dir,
            source=source,
            payload=dict(payload),
            channel=channel,
            tags=list(tags or []),
            corr_id=corr_id,
            parent_id=parent_id,
        )
        self.emitted_events.append(evt)
        return evt

    def metric(self, key: str, value: float, *, ts: Optional[str] = None) -> None:
        ts = ts or _now_iso()
        DB.insert_metric(self.state_dir, key, float(value), ts=ts)
        self.emit_event(
            "metrics.sample",
            {"key": str(key), "value": float(value), "ts": ts},
            tags=["metrics", "consciousness"],
        )


def merged_config(raw: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = dict(raw)
    defaults = {
        "recent_events_limit": 300,
        "recent_broadcast_limit": 300,
        "attention_max_candidates": 12,
        "competition_top_k": 2,
        "competition_reaction_window_secs": 1.5,
        "competition_reaction_min_sources": 2,
        "competition_min_score": 0.15,
    }
    defaults.update(cfg)
    return defaults
