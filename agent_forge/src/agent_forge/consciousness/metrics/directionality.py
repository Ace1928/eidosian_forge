from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from .ignition_trace import event_source, parse_iso_utc


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


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
        if pxy_v > 0.0 and px_v > 0.0 and py_v > 0.0:
            mi += pxy_v * math.log2(pxy_v / (px_v * py_v))
    return _clamp01(mi)


def directionality_asymmetry(
    events: Sequence[Mapping[str, Any]],
    *,
    max_pairs: int = 20,
) -> dict[str, Any]:
    rows = list(events)
    windows: dict[int, set[str]] = defaultdict(set)
    fallback = 0
    for evt in rows:
        src = event_source(evt)
        if not src:
            continue
        parsed = parse_iso_utc(evt.get("ts"))
        if parsed is None:
            w = fallback // 12
            fallback += 1
        else:
            w = int(parsed.timestamp())
        windows[w].add(src)

    if not windows:
        return {
            "window_count": 0,
            "pair_count": 0,
            "mean_abs_asymmetry": 0.0,
            "pairs": [],
        }

    keys = sorted(windows.keys())
    sources = sorted({s for ws in windows.values() for s in ws})
    if len(sources) <= 1:
        return {
            "window_count": len(keys),
            "pair_count": 0,
            "mean_abs_asymmetry": 0.0,
            "pairs": [],
        }

    vectors: dict[str, list[int]] = {src: [] for src in sources}
    for key in keys:
        present = windows[key]
        for src in sources:
            vectors[src].append(1 if src in present else 0)

    pairs: list[dict[str, Any]] = []
    for i, src_a in enumerate(sources):
        for src_b in sources[i + 1 :]:
            a = vectors[src_a]
            b = vectors[src_b]
            if len(a) <= 1 or len(b) <= 1:
                continue
            mi_ab = _binary_mi(a[:-1], b[1:])
            mi_ba = _binary_mi(b[:-1], a[1:])
            asym_ab = mi_ab - mi_ba
            pairs.append(
                {
                    "src": src_a,
                    "dst": src_b,
                    "forward_mi": round(mi_ab, 6),
                    "reverse_mi": round(mi_ba, 6),
                    "asymmetry": round(float(asym_ab), 6),
                    "abs_asymmetry": round(abs(float(asym_ab)), 6),
                }
            )

    pairs.sort(key=lambda row: row["abs_asymmetry"], reverse=True)
    limited = pairs[: max(0, int(max_pairs))]
    mean_abs = sum(row["abs_asymmetry"] for row in limited) / len(limited) if limited else 0.0
    return {
        "window_count": len(keys),
        "pair_count": len(pairs),
        "mean_abs_asymmetry": round(float(mean_abs), 6),
        "pairs": limited,
    }
