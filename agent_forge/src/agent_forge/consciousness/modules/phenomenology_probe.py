from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from ..types import TickContext, WorkspacePayload, clamp01, normalize_workspace_payload


def _parse_iso(ts: Any) -> datetime | None:
    if not isinstance(ts, str) or not ts:
        return None
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


def _event_data(evt: Mapping[str, Any]) -> Mapping[str, Any]:
    return evt.get("data") if isinstance(evt.get("data"), Mapping) else {}


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _window_events(events: Sequence[Mapping[str, Any]], now: datetime, window_secs: float) -> list[dict[str, Any]]:
    cutoff = now - timedelta(seconds=max(0.0, float(window_secs)))
    out: list[dict[str, Any]] = []
    for evt in events:
        ts = _parse_iso(evt.get("ts"))
        if ts is None:
            continue
        if ts >= cutoff:
            out.append(dict(evt))
    return out


def _winner_ids(events: Sequence[Mapping[str, Any]]) -> set[str]:
    out: set[str] = set()
    for evt in events:
        if str(evt.get("type") or "") != "workspace.broadcast":
            continue
        data = _event_data(evt)
        payload = data.get("payload") if isinstance(data.get("payload"), Mapping) else {}
        if str(payload.get("kind") or "") != "GW_WINNER":
            continue
        content = payload.get("content") if isinstance(payload.get("content"), Mapping) else {}
        links = payload.get("links") if isinstance(payload.get("links"), Mapping) else {}
        for value in (
            content.get("winner_candidate_id"),
            content.get("candidate_id"),
            links.get("winner_candidate_id"),
            links.get("candidate_id"),
        ):
            text = str(value or "")
            if text:
                out.add(text)
    return out


def _unity_index(events: Sequence[Mapping[str, Any]], trace_threshold: float) -> tuple[float, dict[str, Any]]:
    traces: list[float] = []
    for evt in events:
        etype = str(evt.get("type") or "")
        data = _event_data(evt)
        if etype == "gw.reaction_trace":
            traces.append(clamp01(data.get("trace_strength"), default=0.0))
        elif etype == "metrics.sample" and str(data.get("key") or "") == "consciousness.ignition.trace_strength":
            traces.append(clamp01(data.get("value"), default=0.0))

    winners = len(_winner_ids(events))
    if not traces:
        ignitions = sum(1 for evt in events if str(evt.get("type") or "") == "gw.ignite")
        unity = 1.0 if ignitions > 0 else 0.0
        return (
            unity,
            {
                "trace_events": 0,
                "strong_trace_events": 0,
                "winner_count": winners,
                "ignition_events": ignitions,
                "trace_threshold": round(float(trace_threshold), 6),
            },
        )

    strong = sum(1 for value in traces if value >= trace_threshold)
    unity = clamp01(strong / len(traces), default=0.0)
    return (
        unity,
        {
            "trace_events": len(traces),
            "strong_trace_events": strong,
            "winner_count": winners,
            "ignition_events": sum(1 for evt in events if str(evt.get("type") or "") == "gw.ignite"),
            "trace_threshold": round(float(trace_threshold), 6),
        },
    )


def _working_set_sequences(events: Sequence[Mapping[str, Any]]) -> list[set[str]]:
    seq: list[set[str]] = []
    for evt in events:
        if str(evt.get("type") or "") != "wm.state":
            continue
        data = _event_data(evt)
        items = data.get("items") if isinstance(data.get("items"), list) else []
        ids: set[str] = set()
        for item in items:
            if not isinstance(item, Mapping):
                continue
            text = str(item.get("item_id") or "")
            if text:
                ids.add(text)
        seq.append(ids)
    return seq


def _continuity_index(events: Sequence[Mapping[str, Any]]) -> tuple[float, dict[str, Any]]:
    seq = _working_set_sequences(events)
    if len(seq) < 2:
        return (
            0.0,
            {
                "wm_state_count": len(seq),
                "mean_overlap": 0.0,
                "persistence_ratio": 0.0,
                "reentry_ratio": 0.0,
                "thread_length_norm": 0.0,
            },
        )

    overlaps: list[float] = []
    positions: dict[str, list[int]] = {}
    for idx, item_ids in enumerate(seq):
        for item_id in item_ids:
            positions.setdefault(item_id, []).append(idx)
        if idx == 0:
            continue
        prev = seq[idx - 1]
        union = prev | item_ids
        overlap = (len(prev & item_ids) / len(union)) if union else 0.0
        overlaps.append(float(overlap))

    persistence_hits = 0
    reentry_hits = 0
    max_streak = 1
    for pos in positions.values():
        has_persistence = any((b - a) == 1 for a, b in zip(pos, pos[1:]))
        has_reentry = any((b - a) > 1 for a, b in zip(pos, pos[1:]))
        if has_persistence:
            persistence_hits += 1
        if has_reentry:
            reentry_hits += 1

        streak = 1
        best = 1
        for a, b in zip(pos, pos[1:]):
            if (b - a) == 1:
                streak += 1
            else:
                streak = 1
            if streak > best:
                best = streak
        if best > max_streak:
            max_streak = best

    unique_items = max(1, len(positions))
    persistence_ratio = persistence_hits / unique_items
    reentry_ratio = reentry_hits / unique_items
    thread_length_norm = max_streak / max(1, len(seq))
    continuity = clamp01(
        (0.45 * _mean(overlaps)) + (0.35 * persistence_ratio) + (0.20 * (1.0 - reentry_ratio)),
        default=0.0,
    )
    return (
        continuity,
        {
            "wm_state_count": len(seq),
            "mean_overlap": round(_mean(overlaps), 6),
            "persistence_ratio": round(persistence_ratio, 6),
            "reentry_ratio": round(reentry_ratio, 6),
            "thread_length_norm": round(thread_length_norm, 6),
            "unique_item_count": len(positions),
        },
    )


def _nearest_before(points: Sequence[tuple[datetime, float]], pivot: datetime, max_secs: float) -> float | None:
    candidates = [val for ts, val in points if ts <= pivot and (pivot - ts).total_seconds() <= max_secs]
    if not candidates:
        return None
    return float(candidates[-1])


def _nearest_after(points: Sequence[tuple[datetime, float]], pivot: datetime, max_secs: float) -> float | None:
    for ts, val in points:
        delta = (ts - pivot).total_seconds()
        if delta < 0:
            continue
        if delta <= max_secs:
            return float(val)
        break
    return None


def _ownership_index(events: Sequence[Mapping[str, Any]]) -> tuple[float, dict[str, Any]]:
    agency_vals: list[tuple[datetime, float]] = []
    boundary_vals: list[float] = []
    perturb_ts: list[datetime] = []

    for evt in events:
        etype = str(evt.get("type") or "")
        ts = _parse_iso(evt.get("ts"))
        data = _event_data(evt)

        if etype == "self.agency_estimate":
            if ts is None:
                continue
            agency_vals.append((ts, clamp01(data.get("agency_confidence"), default=0.0)))
        elif etype == "self.boundary_estimate":
            boundary_vals.append(clamp01(data.get("boundary_stability"), default=0.0))
        elif etype == "metrics.sample":
            key = str(data.get("key") or "")
            value = clamp01(data.get("value"), default=0.0)
            if key == "consciousness.agency" and ts is not None:
                agency_vals.append((ts, value))
            elif key == "consciousness.boundary_stability":
                boundary_vals.append(value)
        elif etype == "perturb.inject" and ts is not None:
            perturb_ts.append(ts)

    agency_vals.sort(key=lambda x: x[0])
    agency_only = [val for _, val in agency_vals]
    agency_mean = _mean(agency_only)
    boundary_mean = _mean(boundary_vals)

    responses: list[float] = []
    if perturb_ts and agency_vals:
        for ts in perturb_ts:
            before = _nearest_before(agency_vals, ts, max_secs=6.0)
            after = _nearest_after(agency_vals, ts, max_secs=6.0)
            if before is None or after is None:
                continue
            responses.append(clamp01(abs(after - before) / 0.25, default=0.0))
    response_score = _mean(responses) if perturb_ts else 0.5

    ownership = clamp01(
        (0.50 * agency_mean) + (0.30 * boundary_mean) + (0.20 * response_score),
        default=0.0,
    )
    return (
        ownership,
        {
            "agency_mean": round(agency_mean, 6),
            "boundary_mean": round(boundary_mean, 6),
            "perturb_count": len(perturb_ts),
            "perturb_response_score": round(response_score, 6),
            "perturb_response_samples": len(responses),
        },
    )


def _perspective_coherence_index(events: Sequence[Mapping[str, Any]]) -> tuple[float, dict[str, Any]]:
    corr_ids = {str(evt.get("corr_id") or "") for evt in events if str(evt.get("corr_id") or "")}
    winner_ids = _winner_ids(events)

    scores: list[float] = []
    report_count = 0
    linked_count = 0

    for evt in events:
        if str(evt.get("type") or "") != "report.self_report":
            continue
        report_count += 1
        data = _event_data(evt)
        evidence = data.get("evidence_links") if isinstance(data.get("evidence_links"), Mapping) else {}
        groundedness = clamp01(data.get("groundedness"), default=0.0)
        summary = data.get("summary") if isinstance(data.get("summary"), Mapping) else {}

        values = [str(v or "") for v in evidence.values() if str(v or "")]
        if values:
            linked_count += 1
            resolved = sum(1 for value in values if value in corr_ids)
            score = resolved / max(1, len(values))
        else:
            score = groundedness

        winner_candidate_id = str(summary.get("winner_candidate_id") or "")
        if winner_candidate_id:
            if winner_candidate_id in winner_ids:
                score = min(1.0, score + 0.08)
            else:
                score *= 0.85

        scores.append(clamp01(score, default=0.0))

    return (
        clamp01(_mean(scores), default=0.0),
        {
            "report_count": report_count,
            "reports_with_links": linked_count,
            "known_winner_count": len(winner_ids),
            "corr_id_count": len(corr_ids),
        },
    )


def _dream_likeness_index(events: Sequence[Mapping[str, Any]]) -> tuple[float, dict[str, Any]]:
    simulated = 0
    real = 0
    meta_modes: list[str] = []
    groundedness_vals: list[float] = []

    for evt in events:
        etype = str(evt.get("type") or "")
        data = _event_data(evt)
        if etype == "sense.simulated_percept":
            simulated += 1
        elif etype == "sense.percept":
            real += 1
        elif etype == "meta.state_estimate":
            mode = str(data.get("mode") or "")
            if mode:
                meta_modes.append(mode)
        elif etype == "report.self_report":
            groundedness_vals.append(clamp01(data.get("groundedness"), default=0.0))
        elif etype == "metrics.sample" and str(data.get("key") or "") == "consciousness.report.groundedness":
            groundedness_vals.append(clamp01(data.get("value"), default=0.0))

    total = simulated + real
    sim_frac = (simulated / total) if total > 0 else 0.0
    mode_sim_frac = sum(1 for mode in meta_modes if mode == "simulated") / len(meta_modes) if meta_modes else 0.0
    ground_mean = _mean(groundedness_vals) if groundedness_vals else 0.5
    mode_support = min(sim_frac, mode_sim_frac)

    dream = clamp01(
        (0.55 * sim_frac) + (0.25 * mode_support) + (0.20 * (1.0 - ground_mean)),
        default=0.0,
    )
    return (
        dream,
        {
            "simulated_percepts": simulated,
            "real_percepts": real,
            "simulated_fraction": round(sim_frac, 6),
            "mode_simulated_fraction": round(mode_sim_frac, 6),
            "groundedness_mean": round(ground_mean, 6),
        },
    )


class PhenomenologyProbeModule:
    name = "phenomenology_probe"

    def tick(self, ctx: TickContext) -> None:
        scan_limit = max(120, int(ctx.config.get("phenom_scan_events", 900)))
        window_secs = max(2.0, float(ctx.config.get("phenom_window_seconds", 16.0)))
        emit_interval = max(0.2, float(ctx.config.get("phenom_emit_interval_secs", 3.0)))
        emit_delta = clamp01(ctx.config.get("phenom_emit_delta_threshold"), default=0.05)
        trace_threshold = clamp01(ctx.config.get("phenom_unity_trace_threshold"), default=0.45)
        broadcast_enable = bool(ctx.config.get("phenom_broadcast_enable", True))
        broadcast_min_confidence = clamp01(ctx.config.get("phenom_broadcast_min_confidence"), default=0.25)

        state = ctx.module_state(
            self.name,
            defaults={
                "last_emit_ts": "",
                "last_snapshot": {},
            },
        )

        all_events = list(ctx.recent_events)[-scan_limit:] + list(ctx.emitted_events)
        windowed = _window_events(all_events, ctx.now, window_secs)
        if not windowed:
            return

        unity_index, unity_evidence = _unity_index(windowed, trace_threshold)
        continuity_index, continuity_evidence = _continuity_index(windowed)
        ownership_index, ownership_evidence = _ownership_index(windowed)
        perspective_index, perspective_evidence = _perspective_coherence_index(windowed)
        dream_index, dream_evidence = _dream_likeness_index(windowed)

        snapshot = {
            "window": {
                "seconds": round(window_secs, 6),
                "event_count": len(windowed),
            },
            "unity_index": round(unity_index, 6),
            "continuity_index": round(continuity_index, 6),
            "ownership_index": round(ownership_index, 6),
            "perspective_coherence_index": round(perspective_index, 6),
            "dream_likeness_index": round(dream_index, 6),
            "evidence": {
                **unity_evidence,
                **continuity_evidence,
                **ownership_evidence,
                **perspective_evidence,
                **dream_evidence,
            },
            "probe_version": 1,
        }

        last_ts = _parse_iso(state.get("last_emit_ts"))
        since_last = (ctx.now - last_ts).total_seconds() if last_ts else 10_000.0
        last_snapshot = state.get("last_snapshot") if isinstance(state.get("last_snapshot"), Mapping) else {}

        max_delta = 0.0
        for key in (
            "unity_index",
            "continuity_index",
            "ownership_index",
            "perspective_coherence_index",
            "dream_likeness_index",
        ):
            prev = clamp01(last_snapshot.get(key), default=0.0)
            cur = clamp01(snapshot.get(key), default=0.0)
            max_delta = max(max_delta, abs(cur - prev))

        should_emit = bool(last_ts is None or since_last >= emit_interval or max_delta >= emit_delta)
        if not should_emit:
            return

        evt = ctx.emit_event(
            "phenom.snapshot",
            snapshot,
            tags=["consciousness", "phenomenology", "probe"],
        )

        ctx.metric("consciousness.phenom.unity_index", float(snapshot["unity_index"]))
        ctx.metric("consciousness.phenom.continuity_index", float(snapshot["continuity_index"]))
        ctx.metric("consciousness.phenom.ownership_index", float(snapshot["ownership_index"]))
        ctx.metric(
            "consciousness.phenom.perspective_coherence_index",
            float(snapshot["perspective_coherence_index"]),
        )
        ctx.metric("consciousness.phenom.dream_likeness_index", float(snapshot["dream_likeness_index"]))

        if broadcast_enable:
            confidence = clamp01(snapshot["perspective_coherence_index"], default=0.0)
            if confidence >= broadcast_min_confidence:
                payload = WorkspacePayload(
                    kind="PHENOM",
                    source_module=self.name,
                    content={
                        "indices": {
                            "unity_index": snapshot["unity_index"],
                            "continuity_index": snapshot["continuity_index"],
                            "ownership_index": snapshot["ownership_index"],
                            "perspective_coherence_index": snapshot["perspective_coherence_index"],
                            "dream_likeness_index": snapshot["dream_likeness_index"],
                        },
                        "evidence": snapshot["evidence"],
                    },
                    confidence=confidence,
                    salience=max(0.15, snapshot["unity_index"] * 0.8),
                    links={
                        "corr_id": evt.get("corr_id"),
                        "parent_id": evt.get("parent_id"),
                        "memory_ids": [],
                    },
                ).as_dict()
                payload = normalize_workspace_payload(
                    payload,
                    fallback_kind="PHENOM",
                    source_module=self.name,
                )
                ctx.broadcast(
                    self.name,
                    payload,
                    tags=["consciousness", "phenomenology", "broadcast"],
                    corr_id=evt.get("corr_id"),
                    parent_id=evt.get("parent_id"),
                )

        state["last_emit_ts"] = ctx.now.strftime("%Y-%m-%dT%H:%M:%SZ")
        state["last_snapshot"] = dict(snapshot)
