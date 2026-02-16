from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Mapping

from ..types import TickContext, WorkspacePayload, normalize_workspace_payload


def _candidate_score(candidate: Mapping[str, Any]) -> float:
    try:
        return float(candidate.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _parse_iso(ts: str) -> datetime | None:
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:
        return None


class WorkspaceCompetitionModule:
    name = "workspace_competition"

    def __init__(self) -> None:
        self._processed_ids: set[str] = set()

    def _collect_candidates(self, ctx: TickContext) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for evt in ctx.recent_events:
            if evt.get("type") != "attn.candidate":
                continue
            data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
            cid = str(data.get("candidate_id") or "")
            if not cid or cid in self._processed_ids:
                continue
            item = dict(data)
            item["_event"] = dict(evt)
            out.append(item)
        return out

    def _recent_reaction_sources(self, ctx: TickContext, *, window_secs: float) -> set[str]:
        cutoff = ctx.now - timedelta(seconds=window_secs)
        sources: set[str] = set()
        for evt in ctx.recent_broadcasts:
            ts = str(evt.get("ts") or "")
            parsed = _parse_iso(ts)
            if parsed is None or parsed < cutoff:
                continue
            data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
            src = str(data.get("source") or "")
            if src:
                sources.add(src)
        return sources

    def tick(self, ctx: TickContext) -> None:
        top_k = int(ctx.config.get("competition_top_k", 2))
        min_score = float(ctx.config.get("competition_min_score", 0.15))
        reaction_window = float(ctx.config.get("competition_reaction_window_secs", 1.5))
        reaction_min_sources = int(ctx.config.get("competition_reaction_min_sources", 2))

        candidates = self._collect_candidates(ctx)
        if not candidates:
            return

        ranked = sorted(candidates, key=_candidate_score, reverse=True)
        winners = [c for c in ranked if _candidate_score(c) >= min_score][:top_k]
        winner_ids = [str(c.get("candidate_id")) for c in winners]

        ctx.emit_event(
            "gw.competition",
            {
                "candidate_count": len(candidates),
                "winner_count": len(winners),
                "winner_ids": winner_ids,
                "top_score": _candidate_score(ranked[0]) if ranked else 0.0,
            },
            tags=["consciousness", "competition"],
        )

        for winner in winners:
            payload = WorkspacePayload(
                kind="GW_WINNER",
                source_module="workspace_competition",
                content={
                    "candidate_id": winner.get("candidate_id"),
                    "source_event_type": winner.get("source_event_type"),
                    "source_module": winner.get("source_module"),
                    "reason": "top_ranked_competition_winner",
                    "score": winner.get("score"),
                },
                confidence=float(winner.get("confidence", 0.5)),
                salience=float(winner.get("salience", 0.5)),
                links=dict(winner.get("links") or {}),
            ).as_dict()
            payload = normalize_workspace_payload(payload, fallback_kind="GW_WINNER", source_module="workspace_competition")
            links = payload.get("links") if isinstance(payload.get("links"), Mapping) else {}
            corr_id = links.get("corr_id")
            parent_id = links.get("parent_id")
            ctx.broadcast(
                "workspace_competition",
                payload,
                tags=["consciousness", "gw", "winner"],
                corr_id=str(corr_id) if corr_id else None,
                parent_id=str(parent_id) if parent_id else None,
            )
            self._processed_ids.add(str(winner.get("candidate_id")))

        reaction_sources = self._recent_reaction_sources(ctx, window_secs=reaction_window)
        if winners and len(reaction_sources) >= reaction_min_sources:
            ctx.emit_event(
                "gw.ignite",
                {
                    "winner_ids": winner_ids,
                    "reaction_sources": sorted(reaction_sources),
                    "reaction_source_count": len(reaction_sources),
                    "reaction_window_secs": reaction_window,
                },
                tags=["consciousness", "gw", "ignite"],
            )

        if winners:
            ctx.metric("consciousness.gw.winner_count", float(len(winners)))
