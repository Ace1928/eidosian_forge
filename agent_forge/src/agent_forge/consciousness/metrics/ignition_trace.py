from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence


def _clamp01(value: Any, *, default: float = 0.0) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return float(default)
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return float(val)


def parse_iso_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def event_source(evt: Mapping[str, Any]) -> str:
    etype = str(evt.get("type") or "")
    data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
    if etype == "workspace.broadcast":
        src = str(data.get("source") or "")
        if src:
            return src
    src = str(data.get("source_module") or data.get("source") or "")
    if src:
        return src
    if "." in etype:
        return etype.split(".", 1)[0]
    return etype or "unknown"


def _payload(evt: Mapping[str, Any]) -> Mapping[str, Any]:
    data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
    return data.get("payload") if isinstance(data.get("payload"), Mapping) else {}


def candidate_references(evt: Mapping[str, Any]) -> set[str]:
    refs: set[str] = set()
    data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
    payload = _payload(evt)
    links = payload.get("links") if isinstance(payload.get("links"), Mapping) else {}
    content = payload.get("content") if isinstance(payload.get("content"), Mapping) else {}
    summary = content.get("summary") if isinstance(content.get("summary"), Mapping) else {}

    for raw in (
        data.get("candidate_id"),
        data.get("winner_candidate_id"),
        links.get("candidate_id"),
        links.get("winner_candidate_id"),
        content.get("candidate_id"),
        content.get("winner_candidate_id"),
        summary.get("winner_candidate_id"),
    ):
        value = str(raw or "")
        if value:
            refs.add(value)
    return refs


def event_references_winner(
    evt: Mapping[str, Any],
    *,
    winner_candidate_id: str,
    winner_corr_id: str,
    winner_parent_id: str,
) -> bool:
    winner_candidate_id = str(winner_candidate_id or "")
    winner_corr_id = str(winner_corr_id or "")
    winner_parent_id = str(winner_parent_id or "")

    evt_corr = str(evt.get("corr_id") or "")
    evt_parent = str(evt.get("parent_id") or "")
    if winner_corr_id and winner_corr_id in {evt_corr, evt_parent}:
        return True
    if winner_parent_id and winner_parent_id in {evt_corr, evt_parent}:
        return True

    payload = _payload(evt)
    links = payload.get("links") if isinstance(payload.get("links"), Mapping) else {}
    link_corr = str(links.get("corr_id") or "")
    link_parent = str(links.get("parent_id") or "")
    if winner_corr_id and winner_corr_id in {link_corr, link_parent}:
        return True
    if winner_parent_id and winner_parent_id in {link_corr, link_parent}:
        return True

    if winner_candidate_id and winner_candidate_id in candidate_references(evt):
        return True
    return False


def winner_reaction_trace(
    events: Sequence[Mapping[str, Any]],
    *,
    winner_candidate_id: str,
    winner_corr_id: str,
    winner_parent_id: str,
    winner_ts: Any,
    reaction_window_secs: float,
    target_sources: int = 5,
    target_reactions: int = 10,
    max_latency_ms: float = 1500.0,
) -> dict[str, Any]:
    winner_candidate_id = str(winner_candidate_id or "")
    winner_corr_id = str(winner_corr_id or "")
    winner_parent_id = str(winner_parent_id or "")
    window_secs = max(0.0, float(reaction_window_secs))
    winner_time = parse_iso_utc(winner_ts)
    window_end = winner_time + timedelta(seconds=window_secs) if winner_time is not None else None

    filtered: list[Mapping[str, Any]] = []
    for evt in events:
        etype = str(evt.get("type") or "")
        if etype.startswith(("attn.", "gw.", "perturb.")):
            continue
        if etype == "metrics.sample":
            continue
        payload = _payload(evt)
        kind = str(payload.get("kind") or "")
        refs = candidate_references(evt)
        if etype == "workspace.broadcast" and kind == "GW_WINNER":
            if winner_candidate_id and winner_candidate_id in refs:
                continue

        evt_time = parse_iso_utc(evt.get("ts"))
        if winner_time is not None and evt_time is not None and evt_time < winner_time:
            continue
        if window_end is not None and evt_time is not None and evt_time > window_end:
            continue
        if event_references_winner(
            evt,
            winner_candidate_id=winner_candidate_id,
            winner_corr_id=winner_corr_id,
            winner_parent_id=winner_parent_id,
        ):
            filtered.append(evt)

    filtered.sort(key=lambda evt: str(evt.get("ts") or ""))
    sources = sorted({event_source(evt) for evt in filtered if event_source(evt)})
    first_latency_ms: float | None = None
    if winner_time is not None and filtered:
        first_ts = parse_iso_utc(filtered[0].get("ts"))
        if first_ts is not None:
            first_latency_ms = round(max(0.0, (first_ts - winner_time).total_seconds() * 1000.0), 6)

    source_component = min(1.0, len(sources) / max(1, int(target_sources)))
    reaction_component = min(1.0, len(filtered) / max(1, int(target_reactions)))
    latency_component = 0.0
    if first_latency_ms is not None:
        latency_component = 1.0 - min(1.0, float(first_latency_ms) / max(1.0, float(max_latency_ms)))
    trace_strength = _clamp01(
        (0.5 * source_component) + (0.3 * reaction_component) + (0.2 * latency_component),
        default=0.0,
    )
    return {
        "winner_candidate_id": winner_candidate_id,
        "winner_corr_id": winner_corr_id,
        "winner_parent_id": winner_parent_id,
        "reaction_count": len(filtered),
        "reaction_sources": sources,
        "reaction_source_count": len(sources),
        "modules_reacted": sources,
        "time_to_first_reaction_ms": first_latency_ms,
        "source_component": round(float(source_component), 6),
        "reaction_component": round(float(reaction_component), 6),
        "latency_component": round(float(latency_component), 6),
        "trace_strength": round(float(trace_strength), 6),
    }
