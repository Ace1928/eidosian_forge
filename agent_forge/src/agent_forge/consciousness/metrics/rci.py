from __future__ import annotations

import json
import zlib
from collections import Counter
from typing import Any, Mapping, Sequence


def response_complexity(events: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not events:
        return {
            "compression_ratio": 0.0,
            "event_type_diversity": 0.0,
            "source_diversity": 0.0,
            "rci": 0.0,
        }

    raw = "\n".join(json.dumps(e, sort_keys=True, default=str) for e in events).encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    compression_ratio = len(compressed) / max(len(raw), 1)

    types = Counter(str(e.get("type") or "" ) for e in events)
    event_type_diversity = len([k for k in types if k]) / max(len(events), 1)

    sources = set()
    for evt in events:
        data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
        src = data.get("source") if isinstance(data, Mapping) else None
        if src:
            sources.add(str(src))
    source_diversity = len(sources) / max(len(events), 1)

    rci = (0.5 * compression_ratio) + (0.25 * event_type_diversity) + (0.25 * source_diversity)
    return {
        "compression_ratio": round(float(compression_ratio), 6),
        "event_type_diversity": round(float(event_type_diversity), 6),
        "source_diversity": round(float(source_diversity), 6),
        "rci": round(float(rci), 6),
    }
