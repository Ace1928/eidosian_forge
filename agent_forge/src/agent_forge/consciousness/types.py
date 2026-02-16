from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
)

from agent_forge.core import db as DB
from agent_forge.core import events as bus
from agent_forge.core import workspace

if TYPE_CHECKING:
    from .state_store import ModuleStateStore


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


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value)
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _target_match(perturb_target: str, target: str) -> bool:
    if perturb_target in {"*", target}:
        return True
    if target == "*":
        return True
    if perturb_target.endswith(".*"):
        prefix = perturb_target[:-2]
        return bool(prefix) and target.startswith(prefix)
    return False


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


def normalize_workspace_payload(
    payload: Mapping[str, Any], *, fallback_kind: str, source_module: str
) -> Dict[str, Any]:
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

    def tick(self, ctx: "TickContext") -> None: ...


@dataclass
class TickContext:
    state_dir: Path
    config: Mapping[str, Any]
    recent_events: Sequence[Dict[str, Any]]
    recent_broadcasts: Sequence[Dict[str, Any]]
    rng: Any
    beat_count: int = 0
    state_store: Optional["ModuleStateStore"] = None
    active_perturbations: Sequence[Mapping[str, Any]] = field(default_factory=list)
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    emitted_events: list[Dict[str, Any]] = field(default_factory=list)
    _event_type_index: Optional[Dict[str, list[Dict[str, Any]]]] = field(
        default=None, init=False, repr=False
    )
    _broadcast_kind_index: Optional[Dict[str, list[Dict[str, Any]]]] = field(
        default=None, init=False, repr=False
    )
    _broadcast_events: Optional[list[Dict[str, Any]]] = field(
        default=None, init=False, repr=False
    )
    _ephemeral_module_state: Dict[str, Dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )

    def all_events(self) -> list[Dict[str, Any]]:
        return list(self.recent_events) + list(self.emitted_events)

    def _build_indexes(self) -> None:
        if (
            self._event_type_index is not None
            and self._broadcast_kind_index is not None
        ):
            return
        by_type: Dict[str, list[Dict[str, Any]]] = {}
        by_kind: Dict[str, list[Dict[str, Any]]] = {}
        broadcasts: list[Dict[str, Any]] = []
        for evt in self.all_events():
            etype = str(evt.get("type") or "")
            if etype:
                by_type.setdefault(etype, []).append(evt)
            if etype != "workspace.broadcast":
                continue
            broadcasts.append(evt)
            data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
            payload = (
                data.get("payload") if isinstance(data.get("payload"), Mapping) else {}
            )
            kind = str(payload.get("kind") or "")
            if kind:
                by_kind.setdefault(kind, []).append(evt)
        self._event_type_index = by_type
        self._broadcast_kind_index = by_kind
        self._broadcast_events = broadcasts

    def latest_event(self, etype: str) -> Optional[Dict[str, Any]]:
        self._build_indexes()
        if self._event_type_index is None:
            return None
        items = self._event_type_index.get(str(etype)) or []
        return items[-1] if items else None

    def latest_events(self, etype: str, k: int = 1) -> list[Dict[str, Any]]:
        self._build_indexes()
        if self._event_type_index is None:
            return []
        items = self._event_type_index.get(str(etype)) or []
        if k <= 0:
            return []
        return list(items[-k:])

    def latest_broadcast(self, kind: str) -> Optional[Dict[str, Any]]:
        self._build_indexes()
        if self._broadcast_kind_index is None:
            return None
        items = self._broadcast_kind_index.get(str(kind)) or []
        return items[-1] if items else None

    def latest_broadcasts(self, kind: str, k: int = 1) -> list[Dict[str, Any]]:
        self._build_indexes()
        if self._broadcast_kind_index is None:
            return []
        items = self._broadcast_kind_index.get(str(kind)) or []
        if k <= 0:
            return []
        return list(items[-k:])

    def all_broadcasts(self) -> list[Dict[str, Any]]:
        self._build_indexes()
        return list(self._broadcast_events or [])

    def module_state(
        self,
        module_name: str,
        *,
        defaults: Optional[Mapping[str, Any]] = None,
    ) -> MutableMapping[str, Any]:
        if self.state_store is not None:
            ns = self.state_store.namespace(module_name, defaults=defaults)
            # State is mutable in-place; mark dirty on access so post-tick flush persists updates.
            self.state_store.mark_dirty()
            return ns
        key = str(module_name)
        ns = self._ephemeral_module_state.setdefault(key, {})
        if defaults:
            for item_key, value in defaults.items():
                ns.setdefault(item_key, value)
        return ns

    def perturbations_for(self, target: str) -> list[Dict[str, Any]]:
        active: list[Dict[str, Any]] = []
        candidates: list[Mapping[str, Any]] = []
        for row in self.active_perturbations:
            if isinstance(row, Mapping):
                candidates.append(row)
        for evt in self.latest_events("perturb.inject", k=32):
            data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
            if isinstance(data, Mapping):
                candidates.append(data)

        for item in candidates:
            perturb_target = str(item.get("target") or "*")
            if not _target_match(perturb_target, str(target)):
                continue
            ts = _parse_ts(item.get("ts"))
            duration = float(item.get("duration_s") or 0.0)
            if ts is not None and duration > 0.0:
                if self.now > (ts + timedelta(seconds=duration)):
                    continue
            active.append(
                {
                    "id": str(item.get("id") or ""),
                    "kind": str(item.get("kind") or ""),
                    "target": perturb_target,
                    "magnitude": float(item.get("magnitude") or 0.0),
                    "duration_s": duration,
                    "meta": dict(item.get("meta") or {}),
                    "ts": str(item.get("ts") or ""),
                }
            )
        return active

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
        self._event_type_index = None
        self._broadcast_kind_index = None
        self._broadcast_events = None
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
        self._event_type_index = None
        self._broadcast_kind_index = None
        self._broadcast_events = None
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
    merged: Dict[str, Any] = {
        "recent_events_limit": 300,
        "recent_broadcast_limit": 300,
        "state_autosave_interval_secs": 2.0,
        "module_tick_periods": {
            "sense": 1,
            "intero": 2,
            "affect": 2,
            "world_model": 1,
            "memory_bridge": 2,
            "knowledge_bridge": 3,
            "attention": 1,
            "workspace_competition": 1,
            "working_set": 1,
            "policy": 1,
            "self_model_ext": 1,
            "meta": 2,
            "report": 2,
        },
        "disable_modules": [],
        "sense_scan_events": 220,
        "sense_max_percepts_per_tick": 6,
        "sense_emit_threshold": 0.72,
        "intero_drive_alpha": 0.22,
        "intero_broadcast_threshold": 0.45,
        "affect_alpha": 0.25,
        "affect_emit_delta_threshold": 0.04,
        "attention_max_candidates": 12,
        "attention_working_set_boost": 0.12,
        "attention_min_confidence": 0.2,
        "competition_top_k": 2,
        "competition_reaction_window_secs": 1.5,
        "competition_reaction_min_sources": 2,
        "competition_reaction_min_count": 2,
        "competition_min_score": 0.15,
        "competition_cooldown_secs": 2.5,
        "competition_cooldown_override_score": 0.9,
        "working_set_capacity": 7,
        "working_set_decay_half_life_secs": 8.0,
        "working_set_min_salience": 0.08,
        "working_set_emit_interval_secs": 2.0,
        "working_set_scan_broadcasts": 120,
        "policy_emit_broadcast": True,
        "policy_max_actions_per_tick": 1,
        "self_emit_delta_threshold": 0.05,
        "self_observation_window": 120,
        "world_prediction_window": 120,
        "world_error_broadcast_threshold": 0.55,
        "memory_bridge_recall_limit": 4,
        "memory_bridge_broadcast_threshold": 0.58,
        "memory_bridge_status_emit_period_beats": 20,
        "memory_bridge_stats_period_beats": 36,
        "memory_bridge_query_max_tokens": 28,
        "memory_bridge_repeat_cooldown_beats": 2,
        "knowledge_bridge_context_limit": 6,
        "knowledge_bridge_broadcast_threshold": 0.55,
        "knowledge_bridge_status_emit_period_beats": 20,
        "knowledge_bridge_query_max_tokens": 32,
        "knowledge_bridge_repeat_cooldown_beats": 2,
        "meta_emit_delta_threshold": 0.05,
        "meta_observation_window": 160,
        "report_emit_interval_secs": 2.0,
        "report_emit_delta_threshold": 0.08,
        "report_broadcast_min_groundedness": 0.35,
    }
    for key, value in cfg.items():
        if isinstance(merged.get(key), Mapping) and isinstance(value, Mapping):
            base = dict(merged.get(key) or {})
            base.update(value)
            merged[key] = base
            continue
        merged[key] = value
    return merged
