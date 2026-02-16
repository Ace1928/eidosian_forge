from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from eidosian_core import eidosian

from . import events as bus

__all__ = [
    "broadcast",
    "iter_broadcast",
    "summary",
]


def _parse_ts(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def _event_source(evt: Mapping[str, Any]) -> str:
    data = evt.get("data") or {}
    return str(
        data.get("source") or data.get("module") or data.get("origin") or "unknown"
    )


@eidosian()
def broadcast(
    base: str | Path,
    source: str,
    payload: Mapping[str, Any],
    *,
    channel: str = "global",
    tags: Optional[Sequence[str]] = None,
    corr_id: str | None = None,
    parent_id: str | None = None,
) -> Dict[str, Any]:
    data = {"source": source, "channel": channel, "payload": dict(payload)}
    return bus.append(
        base,
        "workspace.broadcast",
        data,
        tags=list(tags or []),
        corr_id=corr_id,
        parent_id=parent_id,
    )


@eidosian()
def iter_broadcast(
    base: str | Path,
    *,
    since: str | None = None,
    limit: int | None = 1000,
) -> List[Dict[str, Any]]:
    events = bus.iter_events(base, since=since, limit=limit)
    return [e for e in events if e.get("type") == "workspace.broadcast"]


def _bucket_events(
    events: Sequence[Mapping[str, Any]], window_seconds: float
) -> Dict[int, List[Mapping[str, Any]]]:
    buckets: Dict[int, List[Mapping[str, Any]]] = {}
    for evt in events:
        ts = _parse_ts(str(evt.get("ts", "")))
        bucket = int(ts.timestamp() // window_seconds)
        buckets.setdefault(bucket, []).append(evt)
    return buckets


def _window_metrics(
    events: Sequence[Mapping[str, Any]], window_seconds: float
) -> List[Dict[str, Any]]:
    buckets = _bucket_events(events, window_seconds)
    windows = []
    for bucket, items in sorted(buckets.items(), key=lambda x: x[0]):
        sources = {_event_source(e) for e in items}
        start = datetime.fromtimestamp(bucket * window_seconds, tz=timezone.utc)
        end = datetime.fromtimestamp((bucket + 1) * window_seconds, tz=timezone.utc)
        windows.append(
            {
                "bucket": bucket,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "event_count": len(items),
                "sources": sorted(sources),
            }
        )
    return windows


def _integration_metrics(windows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not windows:
        return {
            "window_count": 0,
            "unique_sources": [],
            "avg_sources_per_window": 0.0,
            "coherence_ratio": 0.0,
        }
    unique_sources: Set[str] = set()
    total_sources = 0
    for win in windows:
        sources = set(win.get("sources", []))
        unique_sources |= sources
        total_sources += len(sources)
    avg_sources = total_sources / len(windows)
    coherence_ratio = avg_sources / max(len(unique_sources), 1)
    return {
        "window_count": len(windows),
        "unique_sources": sorted(unique_sources),
        "avg_sources_per_window": round(avg_sources, 3),
        "coherence_ratio": round(coherence_ratio, 3),
    }


def _gini(values: Sequence[int]) -> float:
    nums = [int(max(0, v)) for v in values]
    if not nums:
        return 0.0
    n = len(nums)
    total = sum(nums)
    if total <= 0:
        return 0.0
    sorted_nums = sorted(nums)
    weighted = 0
    for idx, value in enumerate(sorted_nums, start=1):
        weighted += idx * value
    g = (2 * weighted) / (n * total) - (n + 1) / n
    return round(float(g), 6)


def _ignition_bursts(
    windows: Sequence[Mapping[str, Any]], min_sources: int
) -> Dict[str, Any]:
    bursts = 0
    max_burst = 0
    current = 0
    for win in windows:
        source_count = len(set(win.get("sources") or []))
        if source_count >= min_sources:
            current += 1
            max_burst = max(max_burst, current)
            continue
        if current > 0:
            bursts += 1
            current = 0
    if current > 0:
        bursts += 1
    return {
        "ignition_burst_count": int(bursts),
        "max_ignition_burst": int(max_burst),
    }


def _source_contributions(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for evt in events:
        src = _event_source(evt)
        counts[src] = int(counts.get(src, 0)) + 1
    by_source = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return {
        "source_contributions": by_source[:20],
        "source_gini": _gini([v for _, v in by_source]),
    }


@eidosian()
def summary(
    base: str | Path,
    *,
    since: str | None = None,
    limit: int | None = 1000,
    window_seconds: float = 1.0,
    min_sources: int = 3,
) -> Dict[str, Any]:
    events = iter_broadcast(base, since=since, limit=limit)
    windows = _window_metrics(events, window_seconds)
    ignitions = [w for w in windows if len(w.get("sources", [])) >= min_sources]
    metrics = _integration_metrics(windows)
    bursts = _ignition_bursts(windows, min_sources=min_sources)
    src = _source_contributions(events)
    return {
        "event_count": len(events),
        "window_seconds": window_seconds,
        "min_sources": min_sources,
        "ignition_count": len(ignitions),
        "ignitions": ignitions[:10],
        "mean_sources_per_window": metrics.get("avg_sources_per_window", 0.0),
        **bursts,
        **src,
        **metrics,
    }
