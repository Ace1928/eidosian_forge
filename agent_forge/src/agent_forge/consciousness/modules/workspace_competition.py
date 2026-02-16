from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping

from ..types import TickContext, WorkspacePayload, clamp01, normalize_workspace_payload


def _candidate_score(candidate: Mapping[str, Any]) -> float:
    try:
        return float(candidate.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _parse_iso(ts: str) -> datetime | None:
    try:
        text = ts
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _event_source(evt: Mapping[str, Any]) -> str:
    etype = str(evt.get("type") or "")
    data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
    if etype == "workspace.broadcast":
        src = str(data.get("source") or "")
        if src:
            return src
    if isinstance(data, Mapping):
        src = str(data.get("source_module") or data.get("source") or "")
        if src:
            return src
    if "." in etype:
        return etype.split(".", 1)[0]
    return etype or "unknown"


def _winner_signature(candidate: Mapping[str, Any]) -> str:
    return (
        f"{candidate.get('source_module','')}|"
        f"{candidate.get('source_event_type','')}|"
        f"{candidate.get('kind','')}"
    )


def _string_links(mapping: Mapping[str, Any]) -> dict[str, str]:
    links = mapping.get("links") if isinstance(mapping.get("links"), Mapping) else {}
    return {
        "corr_id": str(links.get("corr_id") or ""),
        "parent_id": str(links.get("parent_id") or ""),
    }


def _event_references_winner(evt: Mapping[str, Any], winner: Mapping[str, Any]) -> bool:
    candidate_id = str(winner.get("candidate_id") or "")
    winner_links = _string_links(winner)
    evt_corr = str(evt.get("corr_id") or "")
    evt_parent = str(evt.get("parent_id") or "")
    if winner_links["corr_id"] and winner_links["corr_id"] in {evt_corr, evt_parent}:
        return True
    if winner_links["parent_id"] and winner_links["parent_id"] in {
        evt_corr,
        evt_parent,
    }:
        return True

    data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
    if candidate_id and str(data.get("candidate_id") or "") == candidate_id:
        return True
    payload = data.get("payload") if isinstance(data.get("payload"), Mapping) else {}
    content = (
        payload.get("content") if isinstance(payload.get("content"), Mapping) else {}
    )
    if candidate_id and str(content.get("candidate_id") or "") == candidate_id:
        return True

    payload_links = (
        _string_links(payload) if payload else {"corr_id": "", "parent_id": ""}
    )
    if winner_links["corr_id"] and winner_links["corr_id"] in set(
        payload_links.values()
    ):
        return True
    if winner_links["parent_id"] and winner_links["parent_id"] in set(
        payload_links.values()
    ):
        return True
    return False


class WorkspaceCompetitionModule:
    name = "workspace_competition"

    def __init__(self) -> None:
        self._processed_ids: set[str] = set()

    def _collect_candidates(self, ctx: TickContext) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for evt in ctx.latest_events("attn.candidate", k=400):
            data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
            cid = str(data.get("candidate_id") or "")
            if not cid or cid in self._processed_ids:
                continue
            item = dict(data)
            item["_event"] = dict(evt)
            out.append(item)
        return out

    def _window_reaction_sources(
        self, ctx: TickContext, *, window_secs: float
    ) -> set[str]:
        cutoff = ctx.now - timedelta(seconds=window_secs)
        sources: set[str] = set()
        for evt in ctx.all_broadcasts():
            parsed = _parse_iso(str(evt.get("ts") or ""))
            if parsed is None or parsed < cutoff:
                continue
            src = _event_source(evt)
            if src:
                sources.add(src)
        return sources

    def _winner_reaction_trace(
        self, ctx: TickContext, winner: Mapping[str, Any], *, window_secs: float
    ) -> Dict[str, Any]:
        cutoff = ctx.now - timedelta(seconds=window_secs)
        reactions: list[Mapping[str, Any]] = []
        for evt in ctx.all_events():
            etype = str(evt.get("type") or "")
            if etype.startswith(("attn.", "gw.", "perturb.")):
                continue
            if etype == "workspace.broadcast":
                data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
                src = str(data.get("source") or "")
                payload = (
                    data.get("payload")
                    if isinstance(data.get("payload"), Mapping)
                    else {}
                )
                content = (
                    payload.get("content")
                    if isinstance(payload.get("content"), Mapping)
                    else {}
                )
                if (
                    src == "workspace_competition"
                    and str(payload.get("kind") or "") == "GW_WINNER"
                ):
                    if str(content.get("candidate_id") or "") == str(
                        winner.get("candidate_id") or ""
                    ):
                        continue
            parsed = _parse_iso(str(evt.get("ts") or ""))
            if parsed is None or parsed < cutoff:
                continue
            if _event_references_winner(evt, winner):
                reactions.append(evt)

        sources = sorted(
            {_event_source(evt) for evt in reactions if _event_source(evt)}
        )
        first_latency_ms = None
        winner_ts = _parse_iso(str((winner.get("_event") or {}).get("ts") or ""))
        if winner_ts and reactions:
            first_evt_ts = _parse_iso(str(reactions[0].get("ts") or ""))
            if first_evt_ts:
                first_latency_ms = round(
                    (first_evt_ts - winner_ts).total_seconds() * 1000.0, 6
                )

        return {
            "candidate_id": str(winner.get("candidate_id") or ""),
            "reaction_count": len(reactions),
            "reaction_sources": sources,
            "reaction_source_count": len(sources),
            "time_to_first_reaction_ms": first_latency_ms,
        }

    def _apply_cooldown_filter(
        self, ctx: TickContext, ranked: list[Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        cooldown_s = max(0.0, float(ctx.config.get("competition_cooldown_secs", 2.5)))
        override_score = clamp01(
            ctx.config.get("competition_cooldown_override_score"), default=0.9
        )
        if cooldown_s <= 0.0:
            return ranked

        state = ctx.module_state(self.name, defaults={"recent_winner_signatures": []})
        raw_recent = state.get("recent_winner_signatures")
        recent: list[Mapping[str, Any]] = []
        if isinstance(raw_recent, list):
            recent = [row for row in raw_recent if isinstance(row, Mapping)]
        by_sig: dict[str, datetime] = {}
        for row in recent:
            sig = str(row.get("signature") or "")
            parsed = _parse_iso(str(row.get("ts") or ""))
            if sig and parsed is not None:
                by_sig[sig] = parsed

        accepted: list[Dict[str, Any]] = []
        kept_recent: list[Dict[str, Any]] = []
        horizon = ctx.now - timedelta(seconds=max(30.0, cooldown_s * 4.0))
        for sig, ts in by_sig.items():
            if ts >= horizon:
                kept_recent.append(
                    {"signature": sig, "ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ")}
                )

        for candidate in ranked:
            sig = _winner_signature(candidate)
            score = _candidate_score(candidate)
            last_seen = by_sig.get(sig)
            if last_seen is not None:
                age = max(0.0, (ctx.now - last_seen).total_seconds())
                if age < cooldown_s and score < override_score:
                    continue
            accepted.append(candidate)
            kept_recent.append(
                {"signature": sig, "ts": ctx.now.strftime("%Y-%m-%dT%H:%M:%SZ")}
            )

        state["recent_winner_signatures"] = kept_recent[-200:]
        return accepted

    def tick(self, ctx: TickContext) -> None:
        top_k = int(ctx.config.get("competition_top_k", 2))
        min_score = float(ctx.config.get("competition_min_score", 0.15))
        reaction_window = float(ctx.config.get("competition_reaction_window_secs", 1.5))
        reaction_min_sources = int(
            ctx.config.get("competition_reaction_min_sources", 2)
        )
        reaction_min_count = int(ctx.config.get("competition_reaction_min_count", 2))
        drop_winners = bool(ctx.config.get("competition_drop_winners", False))

        perturbations = ctx.perturbations_for(self.name)
        if any(str(p.get("kind") or "") == "drop" for p in perturbations):
            drop_winners = True
        noise_mag = max(
            [
                clamp01(p.get("magnitude"), default=0.0)
                for p in perturbations
                if str(p.get("kind") or "") == "noise"
            ]
            or [0.0]
        )
        clamp_floor = 0.0
        for p in perturbations:
            if str(p.get("kind") or "") == "clamp":
                clamp_floor = max(clamp_floor, clamp01(p.get("magnitude"), default=0.0))
        scramble = any(str(p.get("kind") or "") == "scramble" for p in perturbations)
        delayed = any(str(p.get("kind") or "") == "delay" for p in perturbations)
        if delayed and (ctx.beat_count % 2 == 1):
            return

        candidates = self._collect_candidates(ctx)
        if not candidates:
            return

        if noise_mag > 0.0:
            for candidate in candidates:
                noisy = _candidate_score(candidate) + ctx.rng.uniform(
                    -noise_mag, noise_mag
                )
                candidate["score"] = round(clamp01(noisy, default=0.0), 6)
        if scramble:
            ctx.rng.shuffle(candidates)

        ranked = sorted(candidates, key=_candidate_score, reverse=True)
        ranked = self._apply_cooldown_filter(ctx, ranked)
        winners = [
            c for c in ranked if _candidate_score(c) >= max(min_score, clamp_floor)
        ][:top_k]
        winner_ids = [str(c.get("candidate_id")) for c in winners]

        ctx.emit_event(
            "gw.competition",
            {
                "candidate_count": len(candidates),
                "winner_count": len(winners),
                "winner_ids": winner_ids,
                "top_score": _candidate_score(ranked[0]) if ranked else 0.0,
                "perturbations_active": [
                    str(p.get("kind") or "") for p in perturbations
                ],
            },
            tags=["consciousness", "competition"],
        )

        trace_rows: list[Dict[str, Any]] = []
        ignition_sources: set[str] = set()
        ignition_reaction_count = 0
        for winner in winners:
            if drop_winners:
                continue
            raw_links = (
                winner.get("links")
                if isinstance(winner.get("links"), Mapping)
                else {}
            )
            winner_candidate_id = str(winner.get("candidate_id") or "")
            winner_links = ctx.link(
                parent_id=str(raw_links.get("parent_id") or "") or None,
                corr_id=str(raw_links.get("corr_id") or "") or None,
                candidate_id=winner_candidate_id,
                winner_candidate_id=winner_candidate_id,
                memory_ids=list(raw_links.get("memory_ids") or []),
                raw_links=raw_links,
            )
            payload = WorkspacePayload(
                kind="GW_WINNER",
                source_module="workspace_competition",
                content={
                    "candidate_id": winner_candidate_id,
                    "winner_candidate_id": winner_candidate_id,
                    "source_event_type": winner.get("source_event_type"),
                    "source_module": winner.get("source_module"),
                    "reason": "top_ranked_competition_winner",
                    "score": winner.get("score"),
                },
                confidence=float(winner.get("confidence", 0.5)),
                salience=float(winner.get("salience", 0.5)),
                links=winner_links,
            ).as_dict()
            payload = normalize_workspace_payload(
                payload,
                fallback_kind="GW_WINNER",
                source_module="workspace_competition",
            )
            links = (
                payload.get("links")
                if isinstance(payload.get("links"), Mapping)
                else {}
            )
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

            trace = self._winner_reaction_trace(
                ctx, winner, window_secs=reaction_window
            )
            trace_rows.append(trace)
            ignition_sources.update(set(trace.get("reaction_sources") or []))
            ignition_reaction_count += int(trace.get("reaction_count") or 0)
            ctx.emit_event(
                "gw.reaction_trace",
                {
                    "winner_id": str(winner.get("candidate_id") or ""),
                    "winner_corr_id": str(winner_links.get("corr_id") or ""),
                    **trace,
                    "reaction_window_secs": reaction_window,
                },
                tags=["consciousness", "gw", "reaction_trace"],
                corr_id=str(winner_links.get("corr_id") or "") or None,
                parent_id=str(winner_links.get("parent_id") or "") or None,
            )

        if winners and not drop_winners:
            fallback_sources = self._window_reaction_sources(
                ctx, window_secs=reaction_window
            )
            if not ignition_sources:
                ignition_sources = fallback_sources
            ignite = (
                len(ignition_sources) >= reaction_min_sources
                and max(ignition_reaction_count, len(fallback_sources))
                >= reaction_min_count
            )
            if ignite:
                ctx.emit_event(
                    "gw.ignite",
                    {
                        "winner_ids": winner_ids,
                        "reaction_sources": sorted(ignition_sources),
                        "reaction_source_count": len(ignition_sources),
                        "reaction_count": ignition_reaction_count,
                        "reaction_window_secs": reaction_window,
                        "reaction_traces": trace_rows,
                    },
                    tags=["consciousness", "gw", "ignite"],
                )

        if winners and not drop_winners:
            ctx.metric("consciousness.gw.winner_count", float(len(winners)))
            ctx.metric(
                "consciousness.gw.reaction_count", float(ignition_reaction_count)
            )
