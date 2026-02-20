from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

from ..types import TickContext, WorkspacePayload, clamp01, normalize_workspace_payload

_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:-]+")


def _forge_root() -> Path:
    raw = os.environ.get("EIDOS_FORGE_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parents[5]


def _ensure_knowledge_import_path() -> None:
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
    existing_memory = sys.modules.get("memory_forge")
    if existing_memory is not None and not hasattr(existing_memory, "TieredMemorySystem"):
        sys.modules.pop("memory_forge", None)
    existing_knowledge = sys.modules.get("knowledge_forge")
    if existing_knowledge is not None and not hasattr(existing_knowledge, "KnowledgeMemoryBridge"):
        sys.modules.pop("knowledge_forge", None)


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _query_tokens(parts: list[str], *, max_tokens: int) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for token in _TOKEN_RE.findall(part):
            norm = token.lower().strip()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            out.append(norm)
            if len(out) >= max_tokens:
                return " ".join(out)
    return " ".join(out)


class KnowledgeBridgeModule:
    name = "knowledge_bridge"

    def __init__(
        self,
        *,
        bridge: Any = None,
        memory_dir: str | Path | None = None,
        kb_path: str | Path | None = None,
    ) -> None:
        root = _forge_root()
        self._bridge = bridge
        self.memory_dir = (
            Path(memory_dir).expanduser().resolve()
            if memory_dir is not None
            else Path(
                os.environ.get(
                    "EIDOS_MEMORY_DIR",
                    str(root / "data" / "memory"),
                )
            )
            .expanduser()
            .resolve()
        )
        self.kb_path = (
            Path(kb_path).expanduser().resolve()
            if kb_path is not None
            else Path(
                os.environ.get(
                    "EIDOS_KNOWLEDGE_PATH",
                    str(root / "data" / "kb.json"),
                )
            )
            .expanduser()
            .resolve()
        )

    def _load_bridge(self) -> tuple[Any, str]:
        if self._bridge is not None:
            return self._bridge, ""
        try:
            _ensure_knowledge_import_path()
            from knowledge_forge import KnowledgeMemoryBridge  # type: ignore

            self._bridge = KnowledgeMemoryBridge(
                memory_dir=self.memory_dir,
                kb_path=self.kb_path,
            )
            return self._bridge, ""
        except Exception as exc:  # pragma: no cover - defensive fallback
            return None, str(exc)

    def _build_query(self, ctx: TickContext, *, max_tokens: int) -> str:
        parts: list[str] = []
        memory_state = ctx.module_state("memory_bridge", defaults={})
        parts.append(_to_text(memory_state.get("last_query")))

        for evt in ctx.latest_events("mem.recall", k=20):
            data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
            parts.append(_to_text(data.get("query")))
            parts.append(_to_text(data.get("content")))
            parts.append(_to_text(data.get("namespace")))
            parts.append(_to_text(data.get("tier")))

        for evt in ctx.latest_events("workspace.broadcast", k=90):
            data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
            payload = data.get("payload") if isinstance(data.get("payload"), Mapping) else {}
            kind = _to_text(payload.get("kind")).upper()
            if kind not in {
                "GW_WINNER",
                "REPORT",
                "META",
                "SELF",
                "PLAN",
                "PRED_ERR",
                "MEMORY",
                "MEMORY_META",
            }:
                continue
            parts.append(kind)
            content = payload.get("content") if isinstance(payload.get("content"), Mapping) else {}
            for key in (
                "source_module",
                "source_event_type",
                "candidate_id",
                "mode",
                "query",
                "summary",
                "memory_id",
            ):
                parts.append(_to_text(content.get(key)))

        query = _query_tokens(parts, max_tokens=max_tokens)
        if not query:
            query = "consciousness knowledge integration"
        return query

    def _emit_status(
        self,
        ctx: TickContext,
        *,
        available: bool,
        query: str,
        total_hits: int,
        stats: Mapping[str, Any],
        last_error: str,
    ) -> None:
        payload = {
            "available": bool(available),
            "query": query,
            "total_hits": int(total_hits),
            "memory_dir": str(self.memory_dir),
            "kb_path": str(self.kb_path),
            "stats": dict(stats),
            "last_error": last_error,
        }
        ctx.emit_event(
            "knowledge_bridge.status",
            payload,
            tags=["consciousness", "knowledge_bridge", "status"],
        )
        ctx.metric(
            "consciousness.knowledge_bridge.available",
            1.0 if available else 0.0,
        )

    def tick(self, ctx: TickContext) -> None:
        context_limit = max(1, int(ctx.config.get("knowledge_bridge_context_limit", 6)))
        broadcast_threshold = clamp01(
            ctx.config.get("knowledge_bridge_broadcast_threshold"),
            default=0.55,
        )
        status_period = max(1, int(ctx.config.get("knowledge_bridge_status_emit_period_beats", 20)))
        repeat_cooldown = max(0, int(ctx.config.get("knowledge_bridge_repeat_cooldown_beats", 2)))
        query_tokens = max(8, int(ctx.config.get("knowledge_bridge_query_max_tokens", 32)))

        state = ctx.module_state(
            self.name,
            defaults={
                "last_query": "",
                "last_query_hash": "",
                "last_query_beat": -10_000,
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
                    query="",
                    total_hits=0,
                    stats=state.get("last_stats") if isinstance(state.get("last_stats"), Mapping) else {},
                    last_error="perturbation_drop",
                )
            ctx.metric("consciousness.knowledge_bridge.total_hits", 0.0)
            return

        delay_active = any(str(p.get("kind") or "") == "delay" for p in perturbations)
        if delay_active and (ctx.beat_count % 2 == 1):
            return

        noise_mag = max(
            [clamp01(p.get("magnitude"), default=0.0) for p in perturbations if str(p.get("kind") or "") == "noise"]
            or [0.0]
        )
        clamp_ceiling = 1.0
        for pert in perturbations:
            if str(pert.get("kind") or "") == "clamp":
                clamp_ceiling = min(clamp_ceiling, max(0.1, clamp01(pert.get("magnitude"), default=1.0)))

        bridge, load_error = self._load_bridge()
        query = self._build_query(ctx, max_tokens=query_tokens)
        query_hash = hashlib.sha1(query.encode("utf-8", "replace")).hexdigest()
        previous_hash = _to_text(state.get("last_query_hash"))
        previous_query_beat = _safe_int(state.get("last_query_beat"), default=-10_000)

        stats_payload: dict[str, Any] = {}
        memory_context: list[Mapping[str, Any]] = []
        knowledge_context: list[Mapping[str, Any]] = []
        total_hits = 0
        last_error = load_error

        should_query = True
        if previous_hash == query_hash and (ctx.beat_count - previous_query_beat) < repeat_cooldown:
            should_query = False

        if bridge is not None and should_query:
            try:
                result = bridge.get_memory_knowledge_context(query, max_results=context_limit)
                memory_context = list(result.get("memory_context") or [])
                knowledge_context = list(result.get("knowledge_context") or [])
                total_hits = int(result.get("total_results") or (len(memory_context) + len(knowledge_context)))
                stats_payload = bridge.stats() if hasattr(bridge, "stats") else {}
                if isinstance(stats_payload, Mapping):
                    state["last_stats"] = dict(stats_payload)
                else:
                    stats_payload = {}

                context_evt = ctx.emit_event(
                    "knowledge.context",
                    {
                        "query": query,
                        "total_hits": total_hits,
                        "memory_hits": len(memory_context),
                        "knowledge_hits": len(knowledge_context),
                        "memory_context": memory_context,
                        "knowledge_context": knowledge_context,
                    },
                    tags=["consciousness", "knowledge_bridge", "context"],
                )

                rows: list[dict[str, Any]] = []
                for source_name, entries in (
                    ("memory", memory_context),
                    ("knowledge", knowledge_context),
                ):
                    for entry in entries[:context_limit]:
                        if not isinstance(entry, Mapping):
                            continue
                        score = clamp01(entry.get("score"), default=0.0)
                        salience = clamp01(
                            (0.65 * score) + (0.2 if source_name == "knowledge" else 0.1),
                            default=0.4,
                        )
                        if noise_mag > 0.0:
                            salience = clamp01(
                                salience + ctx.rng.uniform(-noise_mag, noise_mag),
                                default=salience,
                            )
                        salience = min(salience, clamp_ceiling)
                        confidence = clamp01(0.35 + (0.65 * score), default=0.35)
                        entry_id = _to_text(entry.get("id") or entry.get("knowledge_id"))
                        content = _to_text(entry.get("content"))
                        row = {
                            "source": source_name,
                            "entry_id": entry_id,
                            "query": query,
                            "score": round(score, 6),
                            "salience": round(salience, 6),
                            "confidence": round(confidence, 6),
                            "content": content,
                            "tags": list(entry.get("tags") or []),
                            "tier": _to_text(entry.get("tier")),
                            "namespace": _to_text(entry.get("namespace")),
                        }
                        rows.append(row)
                        recall_evt = ctx.emit_event(
                            "knowledge.recall",
                            row,
                            tags=["consciousness", "knowledge_bridge", "recall"],
                            corr_id=context_evt.get("corr_id"),
                            parent_id=context_evt.get("parent_id"),
                        )

                        if salience >= broadcast_threshold:
                            memory_ids = [entry_id] if source_name == "memory" and entry_id else []
                            payload = WorkspacePayload(
                                kind="KNOWLEDGE",
                                source_module=self.name,
                                content={
                                    "query": query,
                                    "source": source_name,
                                    "entry_id": entry_id,
                                    "score": round(score, 6),
                                    "summary": content[:240],
                                    "tags": list(entry.get("tags") or []),
                                },
                                confidence=confidence,
                                salience=salience,
                                links={
                                    "corr_id": recall_evt.get("corr_id"),
                                    "parent_id": recall_evt.get("parent_id"),
                                    "memory_ids": memory_ids,
                                },
                            ).as_dict()
                            payload = normalize_workspace_payload(
                                payload,
                                fallback_kind="KNOWLEDGE",
                                source_module=self.name,
                            )
                            ctx.broadcast(
                                self.name,
                                payload,
                                tags=["consciousness", "knowledge_bridge", "broadcast"],
                                corr_id=recall_evt.get("corr_id"),
                                parent_id=recall_evt.get("parent_id"),
                            )

                total_hits = len(rows)
            except Exception as exc:
                last_error = str(exc)

        state["last_query"] = query
        state["last_query_hash"] = query_hash
        state["last_query_beat"] = int(ctx.beat_count)

        if last_error:
            state["last_error"] = last_error
            state["failure_count"] = _safe_int(state.get("failure_count"), default=0) + 1
        else:
            state["last_error"] = ""

        ctx.metric("consciousness.knowledge_bridge.total_hits", float(total_hits))
        ctx.metric(
            "consciousness.knowledge_bridge.memory_hits",
            float(len(memory_context)),
        )
        ctx.metric(
            "consciousness.knowledge_bridge.knowledge_hits",
            float(len(knowledge_context)),
        )

        if (ctx.beat_count % status_period) == 0 or total_hits > 0 or bool(last_error):
            self._emit_status(
                ctx,
                available=bridge is not None,
                query=query,
                total_hits=total_hits,
                stats=stats_payload
                or (state.get("last_stats") if isinstance(state.get("last_stats"), Mapping) else {}),
                last_error=_to_text(state.get("last_error")),
            )
