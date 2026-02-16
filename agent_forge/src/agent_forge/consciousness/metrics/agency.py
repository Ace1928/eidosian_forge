from __future__ import annotations

from typing import Any, Mapping


def agency_confidence(predicted: Mapping[str, Any], observed: Mapping[str, Any]) -> float:
    if not predicted and not observed:
        return 1.0
    keys = set(predicted) | set(observed)
    if not keys:
        return 1.0
    matches = 0
    for key in keys:
        if predicted.get(key) == observed.get(key):
            matches += 1
    return round(matches / max(len(keys), 1), 6)
