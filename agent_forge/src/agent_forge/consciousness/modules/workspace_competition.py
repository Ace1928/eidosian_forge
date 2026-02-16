from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Mapping, MutableMapping

from ..metrics.ignition_trace import parse_iso_utc, winner_reaction_trace
from ..types import TickContext, WorkspacePayload, clamp01, normalize_workspace_payload


def _candidate_score(candidate: Mapping[str, Any]) -> float:
    try:
        return float(candidate.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _winner_signature(candidate: Mapping[str, Any]) -> str:
    return (
        f"{candidate.get('source_module','')}|"
        f"{candidate.get('source_event_type','')}|"
        f"{candidate.get('kind','')}"
    )


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

    def _load_pending_winners(self, ctx: TickContext) -> list[dict[str, Any]]:
        state = ctx.module_state(
            self.name,
            defaults={"recent_winner_signatures": [], "pending_winners": []},
        )
        raw = state.get("pending_winners")
        if not isinstance(raw, list):
            return []
        out: list[dict[str, Any]] = []
        for row in raw:
            if isinstance(row, Mapping):
                out.append(dict(row))
        return out

    def _save_pending_winners(
        self, ctx: TickContext, rows: list[dict[str, Any]]
    ) -> None:
        state = ctx.module_state(
            self.name,
            defaults={"recent_winner_signatures": [], "pending_winners": []},
        )
        state["pending_winners"] = rows[-300:]

    def _candidate_pool_for_winner(
        self,
        ctx: TickContext,
        *,
        winner_candidate_id: str,
        winner_corr_id: str,
        winner_parent_id: str,
    ) -> list[dict[str, Any]]:
        pool: list[dict[str, Any]] = []
        if winner_corr_id:
            pool.extend(ctx.events_by_corr_id(winner_corr_id))
            pool.extend(ctx.children(winner_corr_id))
        if winner_parent_id:
            pool.extend(ctx.events_by_corr_id(winner_parent_id))
            pool.extend(ctx.children(winner_parent_id))
        if winner_candidate_id:
            pool.extend(ctx.candidate_references(winner_candidate_id))

        if not pool:
            pool = list(ctx.all_events())

        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for evt in pool:
            key = (
                str(evt.get("ts") or ""),
                str(evt.get("type") or ""),
                str(evt.get("corr_id") or ""),
                str(evt.get("parent_id") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(evt)
        return deduped

    def _finalize_pending_winners(
        self,
        ctx: TickContext,
        *,
        reaction_window: float,
        reaction_min_sources: int,
        reaction_min_count: int,
        trace_strength_threshold: float,
        trace_target_sources: int,
        trace_target_reactions: int,
        trace_max_latency_ms: float,
        trace_min_eval_secs: float,
    ) -> dict[str, Any]:
        pending = self._load_pending_winners(ctx)
        if not pending:
            return {"traces": [], "ignitions": []}

        pending_next: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        ignitions: list[dict[str, Any]] = []

        for row in pending:
            winner_id = str(row.get("winner_candidate_id") or "")
            winner_corr_id = str(row.get("winner_corr_id") or "")
            winner_parent_id = str(row.get("winner_parent_id") or "")
            winner_ts = str(row.get("winner_ts") or "")
            winner_beat = int(row.get("winner_beat") or -1)

            if not winner_id:
                continue
            if ctx.beat_count <= winner_beat:
                pending_next.append(row)
                continue

            winner_time = parse_iso_utc(winner_ts)
            elapsed = max(
                0.0,
                (ctx.now - winner_time).total_seconds() if winner_time else reaction_window,
            )

            candidate_pool = self._candidate_pool_for_winner(
                ctx,
                winner_candidate_id=winner_id,
                winner_corr_id=winner_corr_id,
                winner_parent_id=winner_parent_id,
            )
            trace = winner_reaction_trace(
                candidate_pool,
                winner_candidate_id=winner_id,
                winner_corr_id=winner_corr_id,
                winner_parent_id=winner_parent_id,
                winner_ts=winner_ts,
                reaction_window_secs=reaction_window,
                target_sources=trace_target_sources,
                target_reactions=trace_target_reactions,
                max_latency_ms=trace_max_latency_ms,
            )

            reaction_count = int(trace.get("reaction_count") or 0)
            source_count = int(trace.get("reaction_source_count") or 0)
            trace_strength = clamp01(trace.get("trace_strength"), default=0.0)
            ready = (
                elapsed >= max(0.0, trace_min_eval_secs)
                and (reaction_count > 0 or elapsed >= reaction_window)
            )
            if not ready:
                pending_next.append(row)
                continue

            trace_event = {
                "winner_id": winner_id,
                "winner_candidate_id": winner_id,
                "winner_corr_id": winner_corr_id,
                "winner_parent_id": winner_parent_id,
                "winner_beat": winner_beat,
                "reaction_window_secs": reaction_window,
                **trace,
            }
            traces.append(trace_event)
            ctx.emit_event(
                "gw.reaction_trace",
                trace_event,
                tags=["consciousness", "gw", "reaction_trace"],
                corr_id=winner_corr_id or None,
                parent_id=winner_parent_id or None,
            )
            ctx.metric("consciousness.ignition.trace_strength", float(trace_strength))
            ctx.metric("consciousness.gw.reaction_count", float(reaction_count))

            ignite = (
                source_count >= max(0, reaction_min_sources)
                and reaction_count >= max(0, reaction_min_count)
                and trace_strength >= clamp01(trace_strength_threshold, default=0.0)
            )
            if ignite:
                ignition_data = {
                    "winner_ids": [winner_id],
                    "winner_candidate_id": winner_id,
                    "winner_corr_id": winner_corr_id,
                    "winner_parent_id": winner_parent_id,
                    "reaction_sources": list(trace.get("reaction_sources") or []),
                    "reaction_source_count": source_count,
                    "reaction_count": reaction_count,
                    "reaction_window_secs": reaction_window,
                    "reaction_traces": [trace],
                    "trace_strength": trace_strength,
                    "ignition_rule": "winner_linked_trace_v3",
                }
                ignitions.append(ignition_data)
                ctx.emit_event(
                    "gw.ignite",
                    ignition_data,
                    tags=["consciousness", "gw", "ignite"],
                    corr_id=winner_corr_id or None,
                    parent_id=winner_parent_id or None,
                )

        self._save_pending_winners(ctx, pending_next)
        return {"traces": traces, "ignitions": ignitions}

    def _apply_cooldown_filter(
        self, ctx: TickContext, ranked: list[Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        cooldown_s = max(0.0, float(ctx.config.get("competition_cooldown_secs", 2.5)))
        override_score = clamp01(
            ctx.config.get("competition_cooldown_override_score"), default=0.9
        )
        if cooldown_s <= 0.0:
            return ranked

        state = ctx.module_state(
            self.name,
            defaults={"recent_winner_signatures": [], "pending_winners": []},
        )
        raw_recent = state.get("recent_winner_signatures")
        recent: list[Mapping[str, Any]] = []
        if isinstance(raw_recent, list):
            recent = [row for row in raw_recent if isinstance(row, Mapping)]
        by_sig: dict[str, datetime] = {}
        for row in recent:
            sig = str(row.get("signature") or "")
            parsed = parse_iso_utc(str(row.get("ts") or ""))
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

    def _adaptive_state(self, ctx: TickContext) -> MutableMapping[str, Any]:
        return ctx.module_state(
            self.name,
            defaults={
                "recent_winner_signatures": [],
                "pending_winners": [],
                "adaptive": {
                    "min_score_bias": 0.0,
                    "top_k_bias": 0.0,
                    "baseline_trace": 0.45,
                    "seen_trace_keys": [],
                },
            },
        )

    def _apply_adaptive_policy(
        self,
        ctx: TickContext,
        *,
        top_k: int,
        min_score: float,
    ) -> tuple[int, float]:
        if not bool(ctx.config.get("competition_adaptive_enabled", True)):
            return top_k, min_score

        state = self._adaptive_state(ctx)
        adaptive = state.get("adaptive")
        if not isinstance(adaptive, MutableMapping):
            adaptive = {
                "min_score_bias": 0.0,
                "top_k_bias": 0.0,
                "baseline_trace": 0.45,
                "seen_trace_keys": [],
            }
            state["adaptive"] = adaptive

        learning_rate = clamp01(ctx.config.get("competition_adaptive_lr"), default=0.08)
        seen_cap = max(80, int(ctx.config.get("competition_adaptive_seen_cap", 400)))
        seen_keys = (
            list(adaptive.get("seen_trace_keys"))
            if isinstance(adaptive.get("seen_trace_keys"), list)
            else []
        )
        seen_set = {str(x) for x in seen_keys}
        baseline_trace = float(adaptive.get("baseline_trace", 0.45) or 0.45)
        min_score_bias = float(adaptive.get("min_score_bias", 0.0) or 0.0)
        top_k_bias = float(adaptive.get("top_k_bias", 0.0) or 0.0)

        updates = 0
        for evt in ctx.latest_events("gw.reaction_trace", k=64):
            data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
            key = (
                f"{data.get('winner_candidate_id','')}:"
                f"{data.get('winner_beat','')}:"
                f"{data.get('reaction_count','')}"
            )
            if key in seen_set:
                continue
            seen_set.add(key)
            seen_keys.append(key)

            trace_strength = clamp01(data.get("trace_strength"), default=0.0)
            reward = trace_strength - baseline_trace
            baseline_trace = (0.95 * baseline_trace) + (0.05 * trace_strength)

            # Positive reward => can admit broader candidate diversity.
            min_score_bias -= learning_rate * reward * 0.28
            top_k_bias += learning_rate * reward * 1.10
            updates += 1

        # Context-sensitive modulation from intero drives.
        intero = ctx.module_state("intero", defaults={"drives": {}})
        drives = intero.get("drives") if isinstance(intero.get("drives"), Mapping) else {}
        threat = clamp01(drives.get("threat"), default=0.0)
        curiosity = clamp01(drives.get("curiosity"), default=0.0)
        min_score_bias += (0.10 * threat) - (0.06 * curiosity)
        top_k_bias += (0.75 * curiosity) - (0.85 * threat)

        min_score_bias = max(-0.2, min(0.25, min_score_bias))
        top_k_bias = max(-1.5, min(2.5, top_k_bias))
        adaptive["min_score_bias"] = round(min_score_bias, 6)
        adaptive["top_k_bias"] = round(top_k_bias, 6)
        adaptive["baseline_trace"] = round(baseline_trace, 6)
        adaptive["seen_trace_keys"] = seen_keys[-seen_cap:]
        state["adaptive"] = adaptive

        effective_top_k = max(
            1,
            int(round(top_k + top_k_bias)),
        )
        effective_top_k = min(
            int(ctx.config.get("competition_adaptive_max_top_k", 5)),
            effective_top_k,
        )
        effective_min_score = max(
            0.01,
            min(0.99, float(min_score + min_score_bias)),
        )
        if updates > 0:
            ctx.emit_event(
                "gw.policy_update",
                {
                    "effective_top_k": effective_top_k,
                    "effective_min_score": round(effective_min_score, 6),
                    "top_k_bias": round(top_k_bias, 6),
                    "min_score_bias": round(min_score_bias, 6),
                    "baseline_trace": round(baseline_trace, 6),
                    "updates": updates,
                    "threat": round(threat, 6),
                    "curiosity": round(curiosity, 6),
                },
                tags=["consciousness", "competition", "learning"],
            )
            ctx.metric("consciousness.gw.policy.top_k", float(effective_top_k))
            ctx.metric("consciousness.gw.policy.min_score", float(effective_min_score))
        return effective_top_k, effective_min_score

    def tick(self, ctx: TickContext) -> None:
        top_k = int(ctx.config.get("competition_top_k", 2))
        min_score = float(ctx.config.get("competition_min_score", 0.15))
        reaction_window = max(
            0.0, float(ctx.config.get("competition_reaction_window_secs", 1.5))
        )
        reaction_min_sources = max(
            0, int(ctx.config.get("competition_reaction_min_sources", 2))
        )
        reaction_min_count = max(
            0, int(ctx.config.get("competition_reaction_min_count", 2))
        )
        trace_strength_threshold = float(
            ctx.config.get("competition_trace_strength_threshold", 0.45)
        )
        trace_target_sources = max(
            1, int(ctx.config.get("competition_trace_target_sources", 5))
        )
        trace_target_reactions = max(
            1, int(ctx.config.get("competition_trace_target_reactions", 10))
        )
        trace_max_latency_ms = float(
            ctx.config.get("competition_trace_max_latency_ms", 1500.0)
        )
        trace_min_eval_secs = max(
            0.0, float(ctx.config.get("competition_trace_min_eval_secs", 0.0))
        )
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

        finalized = self._finalize_pending_winners(
            ctx,
            reaction_window=reaction_window,
            reaction_min_sources=reaction_min_sources,
            reaction_min_count=reaction_min_count,
            trace_strength_threshold=trace_strength_threshold,
            trace_target_sources=trace_target_sources,
            trace_target_reactions=trace_target_reactions,
            trace_max_latency_ms=trace_max_latency_ms,
            trace_min_eval_secs=trace_min_eval_secs,
        )
        if finalized.get("traces"):
            ctx.metric("consciousness.gw.trace_events", float(len(finalized["traces"])))

        top_k, min_score = self._apply_adaptive_policy(
            ctx,
            top_k=top_k,
            min_score=min_score,
        )

        candidates = self._collect_candidates(ctx)
        if not candidates:
            return

        if noise_mag > 0.0:
            for candidate in candidates:
                noisy = _candidate_score(candidate) + ctx.rng.uniform(-noise_mag, noise_mag)
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
                "perturbations_active": [str(p.get("kind") or "") for p in perturbations],
            },
            tags=["consciousness", "competition"],
        )

        pending_rows = self._load_pending_winners(ctx)
        for winner in winners:
            if drop_winners:
                continue
            raw_links = winner.get("links") if isinstance(winner.get("links"), Mapping) else {}
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
            links = payload.get("links") if isinstance(payload.get("links"), Mapping) else {}
            corr_id = str(links.get("corr_id") or "")
            parent_id = str(links.get("parent_id") or "")
            winner_evt = ctx.broadcast(
                "workspace_competition",
                payload,
                tags=["consciousness", "gw", "winner"],
                corr_id=corr_id or None,
                parent_id=parent_id or None,
            )
            self._processed_ids.add(winner_candidate_id)

            pending_rows.append(
                {
                    "winner_candidate_id": winner_candidate_id,
                    "winner_corr_id": corr_id,
                    "winner_parent_id": parent_id,
                    "winner_ts": str(winner_evt.get("ts") or ""),
                    "winner_beat": ctx.beat_count,
                }
            )
        if pending_rows:
            self._save_pending_winners(ctx, pending_rows)

        if winners and not drop_winners:
            ctx.metric("consciousness.gw.winner_count", float(len(winners)))
