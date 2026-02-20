from __future__ import annotations

import json
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

from .index import EventIndex, build_index
from .linking import (
    canonical_links,
    payload_link_candidates,
)
from .linking import new_corr_id as generate_corr_id

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


def _json_size_bytes(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))
    except Exception:
        return len(str(value).encode("utf-8", errors="ignore"))


def _safe_int(value: Any, *, default: int, minimum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return max(minimum, default)
    return max(minimum, parsed)


def _truncate_text(value: str, *, limit: int, stats: Dict[str, int]) -> str:
    if len(value) <= limit:
        return value
    stats["truncated_strings"] = int(stats.get("truncated_strings", 0)) + 1
    return value[:limit] + "...<truncated>"


def _sanitize_value(
    value: Any,
    *,
    max_depth: int,
    max_items: int,
    max_string_chars: int,
    depth: int,
    seen: set[int],
    stats: Dict[str, int],
) -> Any:
    if depth >= max_depth:
        stats["truncated_depth"] = int(stats.get("truncated_depth", 0)) + 1
        return "<truncated:depth>"

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        return _truncate_text(value, limit=max_string_chars, stats=stats)

    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
        stats["coerced_non_json"] = int(stats.get("coerced_non_json", 0)) + 1
        return _truncate_text(text, limit=max_string_chars, stats=stats)

    if isinstance(value, Mapping):
        value_id = id(value)
        if value_id in seen:
            stats["truncated_cycles"] = int(stats.get("truncated_cycles", 0)) + 1
            return "<truncated:cycle>"
        seen.add(value_id)
        out: Dict[str, Any] = {}
        items = list(value.items())
        for idx, (key, item_value) in enumerate(items):
            if idx >= max_items:
                omitted = len(items) - max_items
                if omitted > 0:
                    stats["truncated_items"] = int(stats.get("truncated_items", 0)) + omitted
                    out["_truncated_items"] = omitted
                break
            key_text = _truncate_text(str(key), limit=max_string_chars, stats=stats)
            out[key_text] = _sanitize_value(
                item_value,
                max_depth=max_depth,
                max_items=max_items,
                max_string_chars=max_string_chars,
                depth=depth + 1,
                seen=seen,
                stats=stats,
            )
        seen.discard(value_id)
        return out

    if isinstance(value, (list, tuple, set)):
        value_id = id(value)
        if value_id in seen:
            stats["truncated_cycles"] = int(stats.get("truncated_cycles", 0)) + 1
            return ["<truncated:cycle>"]
        seen.add(value_id)
        seq = list(value)
        out: list[Any] = []
        for idx, item in enumerate(seq):
            if idx >= max_items:
                omitted = len(seq) - max_items
                if omitted > 0:
                    stats["truncated_items"] = int(stats.get("truncated_items", 0)) + omitted
                    out.append({"_truncated_items": omitted})
                break
            out.append(
                _sanitize_value(
                    item,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_string_chars=max_string_chars,
                    depth=depth + 1,
                    seen=seen,
                    stats=stats,
                )
            )
        seen.discard(value_id)
        return out

    stats["coerced_non_json"] = int(stats.get("coerced_non_json", 0)) + 1
    return _truncate_text(str(value), limit=max_string_chars, stats=stats)


def sanitize_payload_mapping(
    payload: Mapping[str, Any],
    *,
    max_payload_bytes: int,
    max_depth: int,
    max_items: int,
    max_string_chars: int,
    max_rounds: int = 5,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    rounds = 0
    current: Any = dict(payload)
    depth = max(2, int(max_depth))
    items = max(4, int(max_items))
    chars = max(32, int(max_string_chars))
    aggregate_stats: Dict[str, int] = {
        "truncated_strings": 0,
        "truncated_items": 0,
        "truncated_depth": 0,
        "truncated_cycles": 0,
        "coerced_non_json": 0,
    }
    used_fallback = False
    oversize_rounds = 0

    while True:
        local_stats: Dict[str, int] = {}
        sanitized = _sanitize_value(
            current,
            max_depth=depth,
            max_items=items,
            max_string_chars=chars,
            depth=0,
            seen=set(),
            stats=local_stats,
        )
        if not isinstance(sanitized, Mapping):
            sanitized = {"value": sanitized}
        for key in aggregate_stats:
            aggregate_stats[key] = int(aggregate_stats[key]) + int(local_stats.get(key, 0))
        payload_size = _json_size_bytes(sanitized)
        if payload_size <= max_payload_bytes:
            break

        oversize_rounds += 1
        rounds += 1
        if rounds >= max_rounds:
            used_fallback = True
            summary = {
                "type": str(type(payload).__name__),
                "keys": [str(k) for k in list(dict(payload).keys())[: max(1, min(items, 8))]],
                "oversize_bytes": payload_size,
            }
            sanitized = {
                "_truncated_payload": True,
                "summary": summary,
            }
            payload_size = _json_size_bytes(sanitized)
            break

        current = sanitized
        depth = max(2, depth - 1)
        items = max(4, items // 2)
        chars = max(32, chars // 2)

    truncated = used_fallback or oversize_rounds > 0 or any(aggregate_stats.values())
    meta: Dict[str, Any] = {
        "truncated": bool(truncated),
        "bytes": int(payload_size),
        "max_payload_bytes": int(max_payload_bytes),
        "oversize_rounds": int(oversize_rounds),
        "used_fallback": bool(used_fallback),
    }
    for key, value in aggregate_stats.items():
        meta[key] = int(value)
    return dict(sanitized), meta


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
    content = payload.get("content") if isinstance(payload.get("content"), Mapping) else {}
    inferred_candidate_id, inferred_winner_candidate_id = payload_link_candidates(payload)
    candidate_id = str(payload.get("candidate_id") or links.get("candidate_id") or inferred_candidate_id or "")
    winner_candidate_id = str(
        payload.get("winner_candidate_id") or links.get("winner_candidate_id") or inferred_winner_candidate_id or ""
    )
    normalized_links = canonical_links(
        links,
        corr_id=str(payload.get("corr_id") or links.get("corr_id") or ""),
        parent_id=str(payload.get("parent_id") or links.get("parent_id") or ""),
        memory_ids=list(links.get("memory_ids") or []),
        candidate_id=candidate_id,
        winner_candidate_id=winner_candidate_id,
    )
    normalized = {
        "kind": str(payload.get("kind") or fallback_kind),
        "ts": str(payload.get("ts") or _now_iso()),
        "id": str(payload.get("id") or uuid.uuid4().hex),
        "source_module": str(payload.get("source_module") or source_module),
        "confidence": clamp01(payload.get("confidence"), default=0.5),
        "salience": clamp01(payload.get("salience"), default=0.5),
        "content": dict(content),
        "links": normalized_links,
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
    _event_type_index: Optional[Dict[str, list[Dict[str, Any]]]] = field(default=None, init=False, repr=False)
    _broadcast_kind_index: Optional[Dict[str, list[Dict[str, Any]]]] = field(default=None, init=False, repr=False)
    _broadcast_events: Optional[list[Dict[str, Any]]] = field(default=None, init=False, repr=False)
    _event_index: Optional[EventIndex] = field(default=None, init=False, repr=False)
    _ephemeral_module_state: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def all_events(self) -> list[Dict[str, Any]]:
        return list(self.recent_events) + list(self.emitted_events)

    def _build_indexes(self) -> None:
        if self._event_type_index is not None and self._broadcast_kind_index is not None:
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
            payload = data.get("payload") if isinstance(data.get("payload"), Mapping) else {}
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

    @property
    def index(self) -> EventIndex:
        if self._event_index is None:
            self._event_index = build_index(self.all_events())
        return self._event_index

    def events_by_corr_id(self, corr_id: str) -> list[Dict[str, Any]]:
        return list(self.index.by_corr_id.get(str(corr_id), []))

    def event(self, event_id: str) -> Optional[Dict[str, Any]]:
        return self.index.by_event_id.get(str(event_id))

    def children(self, parent_id: str) -> list[Dict[str, Any]]:
        return list(self.index.children_by_parent.get(str(parent_id), []))

    def candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        return self.index.candidates_by_id.get(str(candidate_id))

    def winner_for_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        return self.index.winners_by_candidate_id.get(str(candidate_id))

    def candidate_references(self, candidate_id: str) -> list[Dict[str, Any]]:
        return list(self.index.references_by_candidate_id.get(str(candidate_id), []))

    def new_corr_id(self, seed: str | None = None) -> str:
        if seed:
            return generate_corr_id(seed)
        return generate_corr_id()

    def link(
        self,
        *,
        parent_id: str | None = None,
        corr_id: str | None = None,
        candidate_id: str | None = None,
        winner_candidate_id: str | None = None,
        memory_ids: Sequence[str] | None = None,
        raw_links: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_corr_id = str(corr_id or "") or self.new_corr_id(
            seed=f"{self.beat_count}:{candidate_id or winner_candidate_id or ''}"
        )
        return canonical_links(
            raw_links,
            corr_id=resolved_corr_id,
            parent_id=str(parent_id or ""),
            memory_ids=list(memory_ids or []),
            candidate_id=str(candidate_id or ""),
            winner_candidate_id=str(winner_candidate_id or ""),
        )

    def _invalidate_indexes(self) -> None:
        self._event_type_index = None
        self._broadcast_kind_index = None
        self._broadcast_events = None
        self._event_index = None

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

    def _payload_safety_limits(self) -> Dict[str, int]:
        return {
            "max_payload_bytes": _safe_int(
                self.config.get("consciousness_max_payload_bytes"),
                default=16384,
                minimum=1024,
            ),
            "max_depth": _safe_int(
                self.config.get("consciousness_max_depth"),
                default=8,
                minimum=2,
            ),
            "max_items": _safe_int(
                self.config.get("consciousness_max_collection_items"),
                default=64,
                minimum=4,
            ),
            "max_string_chars": _safe_int(
                self.config.get("consciousness_max_string_chars"),
                default=2048,
                minimum=32,
            ),
        }

    def _sanitize_payload(self, payload: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        limits = self._payload_safety_limits()
        return sanitize_payload_mapping(
            payload,
            max_payload_bytes=int(limits["max_payload_bytes"]),
            max_depth=int(limits["max_depth"]),
            max_items=int(limits["max_items"]),
            max_string_chars=int(limits["max_string_chars"]),
        )

    def _emit_payload_truncation(
        self,
        *,
        source_type: str,
        source_name: str,
        source_event_type: str,
        corr_id: str,
        parent_id: str | None,
        channel: str | None = None,
        meta: Mapping[str, Any],
    ) -> None:
        if not bool(self.config.get("consciousness_payload_truncation_event", True)):
            return
        if source_event_type == "consciousness.payload_truncated":
            return
        event_data: Dict[str, Any] = {
            "source_type": str(source_type),
            "source_name": str(source_name),
            "source_event_type": str(source_event_type),
            "channel": str(channel or ""),
            "meta": dict(meta),
        }
        evt = bus.append(
            self.state_dir,
            "consciousness.payload_truncated",
            event_data,
            tags=["consciousness", "payload_safety"],
            corr_id=corr_id,
            parent_id=parent_id or corr_id,
        )
        self.emitted_events.append(evt)
        self.metric("consciousness.payload_truncated.count", 1.0)
        self._invalidate_indexes()

    def emit_event(
        self,
        etype: str,
        data: Optional[Mapping[str, Any]] = None,
        *,
        tags: Optional[Sequence[str]] = None,
        corr_id: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        data_dict = dict(data or {})
        data_dict, payload_meta = self._sanitize_payload(data_dict)
        links = data_dict.get("links") if isinstance(data_dict.get("links"), Mapping) else {}
        resolved_corr_id = str(corr_id or links.get("corr_id") or "") or self.new_corr_id(
            seed=f"{etype}:{self.beat_count}:{len(self.emitted_events)}"
        )
        resolved_parent_id = str(parent_id or links.get("parent_id") or "")
        evt = bus.append(
            self.state_dir,
            etype,
            data_dict,
            tags=list(tags or []),
            corr_id=resolved_corr_id,
            parent_id=resolved_parent_id or None,
        )
        self.emitted_events.append(evt)
        if bool(payload_meta.get("truncated")):
            self._emit_payload_truncation(
                source_type="event",
                source_name=str(etype),
                source_event_type=str(etype),
                corr_id=resolved_corr_id,
                parent_id=resolved_parent_id or None,
                meta=payload_meta,
            )
        self._invalidate_indexes()
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
        fallback_kind = str(payload.get("kind") or "SIGNAL")
        normalized_payload = normalize_workspace_payload(
            payload,
            fallback_kind=fallback_kind,
            source_module=source,
        )
        payload_links = normalized_payload.get("links") if isinstance(normalized_payload.get("links"), Mapping) else {}
        candidate_id, winner_candidate_id = payload_link_candidates(normalized_payload)
        resolved_corr_id = str(corr_id or payload_links.get("corr_id") or "") or self.new_corr_id(
            seed=f"broadcast:{source}:{fallback_kind}:{self.beat_count}:{len(self.emitted_events)}"
        )
        resolved_parent_id = str(parent_id or payload_links.get("parent_id") or "")
        normalized_payload["links"] = self.link(
            parent_id=resolved_parent_id or None,
            corr_id=resolved_corr_id,
            candidate_id=candidate_id or str(payload_links.get("candidate_id") or ""),
            winner_candidate_id=winner_candidate_id or str(payload_links.get("winner_candidate_id") or ""),
            memory_ids=list(payload_links.get("memory_ids") or []),
            raw_links=payload_links,
        )
        safe_payload, payload_meta = self._sanitize_payload(normalized_payload)
        safe_payload.setdefault("kind", normalized_payload.get("kind", fallback_kind))
        safe_payload.setdefault("ts", normalized_payload.get("ts", _now_iso()))
        safe_payload.setdefault("id", normalized_payload.get("id", uuid.uuid4().hex))
        safe_payload.setdefault("source_module", normalized_payload.get("source_module", source))
        safe_payload["confidence"] = clamp01(
            safe_payload.get("confidence", normalized_payload.get("confidence", 0.5)),
            default=0.5,
        )
        safe_payload["salience"] = clamp01(
            safe_payload.get("salience", normalized_payload.get("salience", 0.5)),
            default=0.5,
        )
        if not isinstance(safe_payload.get("content"), Mapping):
            safe_payload["content"] = dict(normalized_payload.get("content") or {})
        if not isinstance(safe_payload.get("links"), Mapping):
            safe_payload["links"] = dict(normalized_payload.get("links") or {})
        safe_payload["links"] = self.link(
            parent_id=resolved_parent_id or None,
            corr_id=resolved_corr_id,
            candidate_id=str((safe_payload.get("links") or {}).get("candidate_id") or candidate_id or ""),
            winner_candidate_id=str(
                (safe_payload.get("links") or {}).get("winner_candidate_id") or winner_candidate_id or ""
            ),
            memory_ids=list((safe_payload.get("links") or {}).get("memory_ids") or []),
            raw_links=(safe_payload.get("links") if isinstance(safe_payload.get("links"), Mapping) else {}),
        )
        evt = workspace.broadcast(
            self.state_dir,
            source=source,
            payload=dict(safe_payload),
            channel=channel,
            tags=list(tags or []),
            corr_id=resolved_corr_id,
            parent_id=resolved_parent_id or None,
        )
        self.emitted_events.append(evt)
        if bool(payload_meta.get("truncated")):
            self._emit_payload_truncation(
                source_type="broadcast",
                source_name=str(source),
                source_event_type="workspace.broadcast",
                corr_id=resolved_corr_id,
                parent_id=resolved_parent_id or None,
                channel=channel,
                meta=payload_meta,
            )
        self._invalidate_indexes()
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
            "simulation": 2,
            "memory_bridge": 2,
            "knowledge_bridge": 3,
            "attention": 1,
            "workspace_competition": 1,
            "working_set": 1,
            "policy": 1,
            "self_model_ext": 1,
            "meta": 2,
            "report": 2,
            "phenomenology_probe": 3,
            "autotune": 60,
            "experiment_designer": 80,
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
        "attention_adaptive_weights_enabled": True,
        "attention_learning_rate": 0.06,
        "attention_weight_update_threshold": 0.02,
        "attention_weight_min": 0.03,
        "attention_seen_trace_cap": 400,
        "attention_component_retention": 600,
        "competition_top_k": 2,
        "competition_reaction_window_secs": 1.5,
        "competition_reaction_min_sources": 2,
        "competition_reaction_min_count": 2,
        "competition_trace_strength_threshold": 0.45,
        "competition_trace_target_sources": 5,
        "competition_trace_target_reactions": 10,
        "competition_trace_max_latency_ms": 1500.0,
        "competition_trace_min_eval_secs": 0.0,
        "competition_min_score": 0.15,
        "competition_cooldown_secs": 2.5,
        "competition_cooldown_override_score": 0.9,
        "competition_adaptive_enabled": True,
        "competition_adaptive_lr": 0.08,
        "competition_adaptive_seen_cap": 400,
        "competition_adaptive_max_top_k": 5,
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
        "world_error_derivative_threshold": 0.2,
        "world_belief_alpha": 0.22,
        "world_belief_top_k": 8,
        "world_belief_max_features": 256,
        "world_rollout_default_steps": 3,
        "simulation_enable": True,
        "simulation_max_per_tick": 3,
        "simulation_observation_window": 120,
        "simulation_quiet_percepts_threshold": 1,
        "simulation_allow_when_quiet": True,
        "simulation_broadcast_min_confidence": 0.35,
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
        "phenom_scan_events": 900,
        "phenom_window_seconds": 16.0,
        "phenom_emit_interval_secs": 3.0,
        "phenom_emit_delta_threshold": 0.05,
        "phenom_unity_trace_threshold": 0.45,
        "phenom_broadcast_enable": True,
        "phenom_broadcast_min_confidence": 0.25,
        "autotune_enabled": False,
        "autotune_interval_beats": 120,
        "autotune_optimizer": "bayes_pareto",
        "autotune_min_improvement": 0.03,
        "autotune_seed_offset": 1_000_000,
        "autotune_task": "signal_pulse",
        "autotune_trial_warmup_beats": 1,
        "autotune_trial_baseline_seconds": 1.5,
        "autotune_trial_perturb_seconds": 1.0,
        "autotune_trial_recovery_seconds": 1.5,
        "autotune_trial_beat_seconds": 0.2,
        "autotune_persist_trials": True,
        "autotune_bandit_step_scale": 0.2,
        "autotune_bayes_candidate_pool": 14,
        "autotune_bayes_kernel_gamma": 3.5,
        "autotune_bayes_kappa": 0.35,
        "autotune_bayes_exploration": 0.12,
        "autotune_guardrail_max_recent_errors": 2,
        "autotune_guardrail_max_module_errors": 0,
        "autotune_guardrail_max_degraded_ratio": 0.45,
        "autotune_guardrail_max_winner_count": 120,
        "autotune_guardrail_max_trace_violations": 0,
        "autotune_run_red_team": True,
        "autotune_red_team_quick": True,
        "autotune_red_team_max_scenarios": 1,
        "autotune_red_team_seed_offset": 2_000_000,
        "autotune_red_team_min_pass_ratio": 0.75,
        "autotune_red_team_min_robustness": 0.70,
        "autotune_red_team_require_available": True,
        "autotune_red_team_persist": False,
        "autotune_red_team_disable_modules": [],
        "experiment_designer_enabled": True,
        "experiment_designer_interval_beats": 120,
        "experiment_designer_auto_inject": False,
        "experiment_designer_min_trials": 3,
        "experiment_designer_recipe_duration_s": 1.5,
        "experiment_designer_recipe_magnitude": 0.35,
        "experiment_designer_max_recent_errors": 3,
        "consciousness_require_links": False,
        "consciousness_max_payload_bytes": 16384,
        "consciousness_max_depth": 8,
        "consciousness_max_collection_items": 64,
        "consciousness_max_string_chars": 2048,
        "consciousness_payload_truncation_event": True,
        "kernel_watchdog_enabled": True,
        "kernel_watchdog_max_consecutive_errors": 3,
        "kernel_watchdog_quarantine_beats": 20,
    }
    for key, value in cfg.items():
        if isinstance(merged.get(key), Mapping) and isinstance(value, Mapping):
            base = dict(merged.get(key) or {})
            base.update(value)
            merged[key] = base
            continue
        merged[key] = value
    return merged
