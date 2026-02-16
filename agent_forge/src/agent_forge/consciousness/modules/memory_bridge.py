from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

from ..types import TickContext, WorkspacePayload, clamp01, normalize_workspace_payload

_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:-]+")


def _forge_root() -> Path:
    raw = os.environ.get("EIDOS_FORGE_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parents[5]


def _ensure_memory_import_path() -> None:
    root = _forge_root()
    candidates = [
        root / "memory_forge" / "src",
        root / "knowledge_forge" / "src",
        root,
    ]
    for path in candidates:
        text = str(path)
        if path.exists() and text not in sys.path:
            sys.path.insert(0, text)
    existing = sys.modules.get("memory_forge")
    if existing is not None and not hasattr(existing, "TieredMemorySystem"):
        sys.modules.pop("memory_forge", None)


def _enum_value(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    inner = getattr(value, "value", value)
    try:
        text = str(inner)
    except Exception:
        return default
    return text or default


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def _tokenize(parts: list[str], *, max_tokens: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for token in _TOKEN_RE.findall(part):
            norm = token.strip().lower()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            out.append(norm)
            if len(out) >= max_tokens:
                return out
    return out


def _score_match(query: str, content: str) -> float:
    q_tokens = set(_TOKEN_RE.findall(query.lower()))
    if not q_tokens:
        return 0.0
    c_tokens = set(_TOKEN_RE.findall(content.lower()))
    if not c_tokens:
        return 0.0
    overlap = len(q_tokens & c_tokens)
    return clamp01(overlap / max(len(q_tokens), 1), default=0.0)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_memory_stats(stats: Any) -> dict[str, Any]:
    if isinstance(stats, Mapping):
        return dict(stats)
    return {
        "total_memories": _safe_int(getattr(stats, "total_memories", 0), default=0),
        "by_tier": dict(getattr(stats, "by_tier", {}) or {}),
        "by_namespace": dict(getattr(stats, "by_namespace", {}) or {}),
        "by_type": dict(getattr(stats, "by_type", {}) or {}),
        "avg_importance": float(getattr(stats, "avg_importance", 0.0) or 0.0),
        "avg_access_count": float(getattr(stats, "avg_access_count", 0.0) or 0.0),
        "top_tags": list(getattr(stats, "top_tags", []) or []),
    }


def _parse_insights(items: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if isinstance(item, Mapping):
            out.append(
                {
                    "type": _to_text(item.get("insight_type") or item.get("type") or "unknown"),
                    "description": _to_text(item.get("description")),
                    "confidence": clamp01(item.get("confidence"), default=0.0),
                    "evidence": list(item.get("evidence") or []),
                }
            )
            continue
        out.append(
            {
                "type": _to_text(getattr(item, "insight_type", "unknown")),
                "description": _to_text(getattr(item, "description", "")),
                "confidence": clamp01(getattr(item, "confidence", 0.0), default=0.0),
                "evidence": list(getattr(item, "evidence", []) or []),
            }
        )
    return out


class MemoryBridgeModule:
    name = "memory_bridge"

    def __init__(
        self,
        *,
        memory_system: Any = None,
        introspector: Any = None,
        memory_dir: str | Path | None = None,
    ) -> None:
        self._memory_system = memory_system
        self._introspector = introspector
        self.memory_dir = (
            Path(memory_dir).expanduser().resolve()
            if memory_dir is not None
            else Path(
                os.environ.get(
                    "EIDOS_MEMORY_DIR",
                    str(_forge_root() / "data" / "memory"),
                )
            ).expanduser().resolve()
        )

    def _load_memory_system(self) -> tuple[Any, str]:
        if self._memory_system is not None:
            return self._memory_system, ""
        try:
            _ensure_memory_import_path()
            from memory_forge import TieredMemorySystem  # type: ignore

            self._memory_system = TieredMemorySystem(persistence_dir=self.memory_dir)
            return self._memory_system, ""
        except Exception as exc:  # pragma: no cover - defensive runtime fallback
            return None, str(exc)

    def _load_introspector(self) -> tuple[Any, str]:
        if self._introspector is not None:
            return self._introspector, ""
        try:
            _ensure_memory_import_path()
            from memory_forge import MemoryIntrospector  # type: ignore

            self._introspector = MemoryIntrospector(self.memory_dir)
            return self._introspector, ""
        except Exception as exc:  # pragma: no cover - defensive runtime fallback
            return None, str(exc)

    def _build_query(self, ctx: TickContext, *, max_tokens: int) -> str:
        parts: list[str] = []
        for evt in ctx.latest_events("workspace.broadcast", k=140):
            data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
            payload = data.get("payload") if isinstance(data.get("payload"), Mapping) else {}
            kind = _to_text(payload.get("kind")).upper()
            if kind not in {
                "GW_WINNER",
                "WM_STATE",
                "PLAN",
                "SELF",
                "META",
                "REPORT",
                "PRED_ERR",
                "DRIVE",
                "PERCEPT",
            }:
                continue
            parts.append(kind)
            content = payload.get("content") if isinstance(payload.get("content"), Mapping) else {}
            for key in (
                "source_event_type",
                "source_module",
                "candidate_id",
                "action_kind",
                "mode",
                "drive_name",
                "actual_event_type",
                "predicted_event_type",
            ):
                parts.append(_to_text(content.get(key)))
            summary = content.get("summary") if isinstance(content.get("summary"), Mapping) else {}
            for key in ("action_kind", "selected_candidate_id", "winner_candidate_id"):
                parts.append(_to_text(summary.get(key)))

        for evt in ctx.latest_events("sense.percept", k=12):
            data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
            parts.append(_to_text(data.get("source_event_type")))
            parts.append(_to_text(data.get("source_module")))

        if not any(parts):
            parts.extend(["consciousness", "workspace", "agent_forge"])

        tokens = _tokenize(parts, max_tokens=max_tokens)
        return " ".join(tokens)

    def _emit_status(
        self,
        ctx: TickContext,
        *,
        available: bool,
        introspector_available: bool,
        query: str,
        recall_count: int,
        last_error: str,
        stats: Mapping[str, Any] | None,
    ) -> None:
        data = {
            "available": bool(available),
            "introspector_available": bool(introspector_available),
            "memory_dir": str(self.memory_dir),
            "query": query,
            "recall_count": int(recall_count),
            "last_error": last_error,
            "stats": dict(stats or {}),
        }
        ctx.emit_event(
            "memory_bridge.status",
            data,
            tags=["consciousness", "memory_bridge", "status"],
        )
        ctx.metric(
            "consciousness.memory_bridge.available",
            1.0 if available else 0.0,
        )

    def tick(self, ctx: TickContext) -> None:
        recall_limit = max(1, int(ctx.config.get("memory_bridge_recall_limit", 4)))
        broadcast_threshold = clamp01(
            ctx.config.get("memory_bridge_broadcast_threshold"), default=0.58
        )
        status_period = max(
            1, int(ctx.config.get("memory_bridge_status_emit_period_beats", 20))
        )
        stats_period = max(
            1, int(ctx.config.get("memory_bridge_stats_period_beats", 36))
        )
        query_tokens = max(6, int(ctx.config.get("memory_bridge_query_max_tokens", 28)))
        repeat_cooldown = max(
            0, int(ctx.config.get("memory_bridge_repeat_cooldown_beats", 2))
        )

        state = ctx.module_state(
            self.name,
            defaults={
                "last_query": "",
                "last_query_hash": "",
                "last_query_beat": -10_000,
                "last_recall_ids": [],
                "last_error": "",
                "failure_count": 0,
                "last_stats": {},
            },
        )

        perturbations = ctx.perturbations_for(self.name)
        if any(str(p.get("kind") or "") == "drop" for p in perturbations):
            if (ctx.beat_count % status_period) == 0:
                self._emit_status(
                    ctx,
                    available=False,
                    introspector_available=False,
                    query="",
                    recall_count=0,
                    last_error="perturbation_drop",
                    stats=state.get("last_stats") if isinstance(state.get("last_stats"), Mapping) else {},
                )
            ctx.metric("consciousness.memory_bridge.recalls", 0.0)
            return

        delay_active = any(str(p.get("kind") or "") == "delay" for p in perturbations)
        if delay_active and (ctx.beat_count % 2 == 1):
            return

        noise_mag = max(
            [
                clamp01(p.get("magnitude"), default=0.0)
                for p in perturbations
                if str(p.get("kind") or "") == "noise"
            ]
            or [0.0]
        )
        clamp_ceiling = 1.0
        for pert in perturbations:
            if str(pert.get("kind") or "") == "clamp":
                clamp_ceiling = min(clamp_ceiling, max(0.1, clamp01(pert.get("magnitude"), default=1.0)))

        memory_system, memory_err = self._load_memory_system()
        introspector, intro_err = self._load_introspector()
        last_error = memory_err or intro_err

        stats_payload: dict[str, Any] = {}
        insight_count = 0
        if introspector is not None and (ctx.beat_count % stats_period) == 0:
            try:
                stats_obj = introspector.get_stats()
                insights_obj = introspector.analyze_patterns()
                stats_payload = _parse_memory_stats(stats_obj)
                insights = _parse_insights(insights_obj)
                insight_count = len(insights)
                state["last_stats"] = stats_payload
                state["last_insights"] = insights[:6]
                introspection_evt = ctx.emit_event(
                    "mem.introspection",
                    {
                        "stats": stats_payload,
                        "insight_count": insight_count,
                        "insights": insights[:6],
                    },
                    tags=["consciousness", "memory_bridge", "introspection"],
                )
                if insights and max(
                    (clamp01(item.get("confidence"), default=0.0) for item in insights),
                    default=0.0,
                ) >= 0.75:
                    payload = WorkspacePayload(
                        kind="MEMORY_META",
                        source_module=self.name,
                        content={
                            "insight_count": insight_count,
                            "top_insight": insights[0].get("description") if insights else "",
                            "total_memories": stats_payload.get("total_memories"),
                        },
                        confidence=0.75,
                        salience=0.65,
                        links={
                            "corr_id": introspection_evt.get("corr_id"),
                            "parent_id": introspection_evt.get("parent_id"),
                            "memory_ids": [],
                        },
                    ).as_dict()
                    payload = normalize_workspace_payload(
                        payload,
                        fallback_kind="MEMORY_META",
                        source_module=self.name,
                    )
                    ctx.broadcast(
                        self.name,
                        payload,
                        tags=["consciousness", "memory_bridge", "broadcast"],
                        corr_id=introspection_evt.get("corr_id"),
                        parent_id=introspection_evt.get("parent_id"),
                    )
                ctx.metric("consciousness.memory_bridge.insight_count", float(insight_count))
            except Exception as exc:
                last_error = str(exc)

        query = self._build_query(ctx, max_tokens=query_tokens)
        query_hash = hashlib.sha1(query.encode("utf-8", "replace")).hexdigest()
        previous_hash = _to_text(state.get("last_query_hash"))
        previous_query_beat = _safe_int(state.get("last_query_beat"), default=-10_000)

        recall_rows: list[dict[str, Any]] = []
        recalled_ids: list[str] = []
        if memory_system is not None and query:
            should_query = True
            if previous_hash == query_hash and (ctx.beat_count - previous_query_beat) < repeat_cooldown:
                should_query = False

            if should_query:
                try:
                    recalled = memory_system.recall(query, limit=recall_limit)
                    for item in recalled:
                        memory_id = _to_text(getattr(item, "id", ""))
                        if not memory_id:
                            continue
                        content = _to_text(getattr(item, "content", ""))
                        tier = _enum_value(getattr(item, "tier", None))
                        namespace = _enum_value(getattr(item, "namespace", None))
                        importance = clamp01(getattr(item, "importance", 0.5), default=0.5)
                        access_count = max(0, _safe_int(getattr(item, "access_count", 0), default=0))
                        score = _score_match(query, content)
                        novelty = clamp01(1.0 / (1.0 + float(access_count)), default=0.0)

                        salience = clamp01(
                            (0.42 * importance) + (0.36 * score) + (0.22 * novelty),
                            default=0.4,
                        )
                        if noise_mag > 0.0:
                            salience = clamp01(
                                salience + ctx.rng.uniform(-noise_mag, noise_mag),
                                default=salience,
                            )
                        salience = min(salience, clamp_ceiling)
                        confidence = clamp01(0.35 + (0.65 * score), default=0.35)

                        event_data = {
                            "memory_id": memory_id,
                            "query": query,
                            "score": round(score, 6),
                            "novelty": round(novelty, 6),
                            "importance": round(importance, 6),
                            "salience": round(salience, 6),
                            "confidence": round(confidence, 6),
                            "tier": tier,
                            "namespace": namespace,
                            "access_count": access_count,
                            "tags": sorted(list(getattr(item, "tags", set()) or [])),
                            "content": content,
                        }
                        recall_evt = ctx.emit_event(
                            "mem.recall",
                            event_data,
                            tags=["consciousness", "memory_bridge", "recall"],
                        )
                        recall_rows.append(event_data)
                        recalled_ids.append(memory_id)

                        if salience >= broadcast_threshold:
                            payload = WorkspacePayload(
                                kind="MEMORY",
                                source_module=self.name,
                                content={
                                    "memory_id": memory_id,
                                    "query": query,
                                    "score": round(score, 6),
                                    "tier": tier,
                                    "namespace": namespace,
                                    "summary": content[:240],
                                },
                                confidence=confidence,
                                salience=salience,
                                links={
                                    "corr_id": recall_evt.get("corr_id"),
                                    "parent_id": recall_evt.get("parent_id"),
                                    "memory_ids": [memory_id],
                                },
                            ).as_dict()
                            payload = normalize_workspace_payload(
                                payload,
                                fallback_kind="MEMORY",
                                source_module=self.name,
                            )
                            ctx.broadcast(
                                self.name,
                                payload,
                                tags=["consciousness", "memory_bridge", "broadcast"],
                                corr_id=recall_evt.get("corr_id"),
                                parent_id=recall_evt.get("parent_id"),
                            )
                except Exception as exc:
                    last_error = str(exc)
            state["last_query"] = query
            state["last_query_hash"] = query_hash
            state["last_query_beat"] = int(ctx.beat_count)

        state["last_recall_ids"] = recalled_ids[: recall_limit * 3]

        if last_error:
            state["failure_count"] = _safe_int(state.get("failure_count"), default=0) + 1
            state["last_error"] = last_error
        else:
            state["last_error"] = ""

        recalls = float(len(recall_rows))
        ctx.metric("consciousness.memory_bridge.recalls", recalls)

        if (ctx.beat_count % status_period) == 0 or recalls > 0.0 or bool(last_error):
            self._emit_status(
                ctx,
                available=memory_system is not None,
                introspector_available=introspector is not None,
                query=query,
                recall_count=int(len(recall_rows)),
                last_error=_to_text(state.get("last_error")),
                stats=stats_payload
                or (
                    state.get("last_stats")
                    if isinstance(state.get("last_stats"), Mapping)
                    else {}
                ),
            )
