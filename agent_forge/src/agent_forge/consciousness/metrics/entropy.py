from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from .ignition_trace import event_source


def _safe_label(value: Any) -> str:
    text = str(value or "").strip()
    return text


def shannon_entropy(labels: Iterable[str]) -> float:
    counts = Counter(_safe_label(label) for label in labels if _safe_label(label))
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0.0:
            entropy -= p * math.log2(p)
    return float(entropy)


def normalized_entropy(labels: Iterable[str]) -> float:
    counts = Counter(_safe_label(label) for label in labels if _safe_label(label))
    k = len(counts)
    if k <= 1:
        return 0.0
    raw = shannon_entropy(counts.elements())
    return float(raw / math.log2(k))


def event_type_entropy(events: Sequence[Mapping[str, Any]]) -> float:
    return normalized_entropy(str(evt.get("type") or "") for evt in events)


def source_entropy(events: Sequence[Mapping[str, Any]]) -> float:
    return normalized_entropy(event_source(evt) for evt in events)
