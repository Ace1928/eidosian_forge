from __future__ import annotations

import json
import math
import zlib
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from .entropy import event_type_entropy, source_entropy
from .ignition_trace import event_source, parse_iso_utc


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp01(value: Any, default: float = 0.0) -> float:
    val = _safe_float(value, default)
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return float(val)


def _normalized_counter_diversity(counter: Counter[str], total: int) -> float:
    if total <= 0:
        return 0.0
    keys = [k for k in counter.keys() if k]
    return float(len(keys) / total)


def _transition_entropy(types: list[str]) -> float:
    if len(types) <= 1:
        return 0.0
    pairs = [f"{a}->{b}" for a, b in zip(types, types[1:]) if a and b]
    if not pairs:
        return 0.0
    counts = Counter(pairs)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0.0:
            ent -= p * math.log2(p)
    k = len(counts)
    if k <= 1:
        return 0.0
    return float(ent / math.log2(k))


def _burstiness(events: Sequence[Mapping[str, Any]]) -> float:
    if not events:
        return 0.0
    buckets: dict[int, int] = defaultdict(int)
    fallback_idx = 0
    for evt in events:
        parsed = parse_iso_utc(evt.get("ts"))
        if parsed is None:
            bucket = fallback_idx // 10
            fallback_idx += 1
        else:
            bucket = int(parsed.timestamp())
        buckets[bucket] += 1

    counts = list(buckets.values())
    if not counts:
        return 0.0
    mean = sum(counts) / len(counts)
    if mean <= 1e-9:
        return 0.0
    var = sum((c - mean) ** 2 for c in counts) / len(counts)
    fano = var / mean
    # Normalize: Fano~1 near Poisson baseline, >1 bursty. Map into [0,1].
    return _clamp01(fano / (fano + 1.0), default=0.0)


def _lag_repetition(types: list[str]) -> float:
    if len(types) <= 1:
        return 0.0
    same = 0
    total = 0
    for a, b in zip(types, types[1:]):
        if not a or not b:
            continue
        total += 1
        if a == b:
            same += 1
    if total <= 0:
        return 0.0
    return float(same / total)


def _binary_mi(xs: list[int], ys: list[int]) -> float:
    n = min(len(xs), len(ys))
    if n <= 0:
        return 0.0
    pxy = Counter((xs[i], ys[i]) for i in range(n))
    px = Counter(xs[:n])
    py = Counter(ys[:n])
    mi = 0.0
    for (x, y), cxy in pxy.items():
        pxy_v = cxy / n
        px_v = px[x] / n
        py_v = py[y] / n
        if pxy_v > 0 and px_v > 0 and py_v > 0:
            mi += pxy_v * math.log2(pxy_v / (px_v * py_v))
    # Binary MI max is 1 bit.
    return _clamp01(mi, default=0.0)


def _integration_proxy(events: Sequence[Mapping[str, Any]]) -> float:
    if len(events) <= 2:
        return 0.0
    windows: dict[int, set[str]] = defaultdict(set)
    fallback = 0
    for evt in events:
        src = event_source(evt)
        if not src:
            continue
        parsed = parse_iso_utc(evt.get("ts"))
        if parsed is None:
            window_idx = fallback // 12
            fallback += 1
        else:
            window_idx = int(parsed.timestamp())
        windows[window_idx].add(src)

    if not windows:
        return 0.0
    all_sources = sorted({src for vals in windows.values() for src in vals})
    if len(all_sources) <= 1:
        return 0.0

    ordered_keys = sorted(windows.keys())
    vectors: dict[str, list[int]] = {src: [] for src in all_sources}
    for key in ordered_keys:
        present = windows[key]
        for src in all_sources:
            vectors[src].append(1 if src in present else 0)

    mi_vals: list[float] = []
    for i, src_a in enumerate(all_sources):
        for src_b in all_sources[i + 1 :]:
            mi_vals.append(_binary_mi(vectors[src_a], vectors[src_b]))
    if not mi_vals:
        return 0.0
    return float(sum(mi_vals) / len(mi_vals))


def response_complexity(events: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not events:
        return {
            "compression_ratio": 0.0,
            "event_type_diversity": 0.0,
            "source_diversity": 0.0,
            "event_type_entropy": 0.0,
            "source_entropy": 0.0,
            "transition_entropy": 0.0,
            "burstiness": 0.0,
            "integration_proxy": 0.0,
            "lag1_repetition": 0.0,
            "rci": 0.0,
            "rci_v2": 0.0,
        }

    raw = "\n".join(json.dumps(e, sort_keys=True, default=str) for e in events).encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    compression_ratio = len(compressed) / max(len(raw), 1)

    types = [str(e.get("type") or "") for e in events]
    type_counter = Counter(types)
    event_type_diversity = _normalized_counter_diversity(type_counter, len(events))

    sources = [event_source(evt) for evt in events]
    src_counter = Counter(src for src in sources if src)
    source_diversity = _normalized_counter_diversity(src_counter, len(events))

    type_entropy = event_type_entropy(events)
    src_entropy = source_entropy(events)
    transition_entropy = _transition_entropy(types)
    burstiness = _burstiness(events)
    integration_proxy = _integration_proxy(events)
    lag1_repetition = _lag_repetition(types)

    # Backward-compatible v1 RCI.
    rci = (0.5 * compression_ratio) + (0.25 * event_type_diversity) + (0.25 * source_diversity)
    # v2 includes entropy/temporal/integration features.
    rci_v2 = (
        (0.20 * compression_ratio)
        + (0.10 * event_type_diversity)
        + (0.10 * source_diversity)
        + (0.15 * type_entropy)
        + (0.15 * src_entropy)
        + (0.10 * transition_entropy)
        + (0.10 * burstiness)
        + (0.10 * integration_proxy)
    )

    return {
        "compression_ratio": round(float(compression_ratio), 6),
        "event_type_diversity": round(float(event_type_diversity), 6),
        "source_diversity": round(float(source_diversity), 6),
        "event_type_entropy": round(float(type_entropy), 6),
        "source_entropy": round(float(src_entropy), 6),
        "transition_entropy": round(float(transition_entropy), 6),
        "burstiness": round(float(burstiness), 6),
        "integration_proxy": round(float(integration_proxy), 6),
        "lag1_repetition": round(float(lag1_repetition), 6),
        "rci": round(float(rci), 6),
        "rci_v2": round(float(rci_v2), 6),
    }
