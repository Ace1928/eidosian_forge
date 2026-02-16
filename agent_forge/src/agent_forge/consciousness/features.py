from __future__ import annotations

from typing import Any, Mapping

from .metrics.ignition_trace import event_source


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _guess_kind_from_type(etype: str) -> str:
    if etype.startswith("sense."):
        return "PERCEPT"
    if etype.startswith("intero."):
        return "DRIVE"
    if etype.startswith("policy."):
        return "PLAN"
    if etype.startswith("self."):
        return "SELF"
    if etype.startswith("meta."):
        return "META"
    if etype.startswith("report."):
        return "REPORT"
    if etype.startswith("world."):
        return "WORLD"
    return "SIGNAL"


def extract_event_features(evt: Mapping[str, Any]) -> dict[str, float]:
    etype = str(evt.get("type") or "")
    data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
    payload = data.get("payload") if isinstance(data.get("payload"), Mapping) else {}
    content = payload.get("content") if isinstance(payload.get("content"), Mapping) else {}

    source = event_source(evt)
    kind = str(payload.get("kind") or _guess_kind_from_type(etype))

    feats: dict[str, float] = {}
    if etype:
        feats[f"etype:{etype}"] = 1.0
        prefix = etype.split(".", 1)[0]
        feats[f"etype_prefix:{prefix}"] = 1.0
    if source:
        feats[f"source:{source}"] = 1.0
    if kind:
        feats[f"kind:{kind}"] = 1.0

    salience = _safe_float(payload.get("salience"))
    confidence = _safe_float(payload.get("confidence"))
    if salience is not None:
        feats["signal:salience"] = max(0.0, min(1.0, salience))
    if confidence is not None:
        feats["signal:confidence"] = max(0.0, min(1.0, confidence))

    numeric_candidates = {
        "novelty": data.get("novelty"),
        "drive_strength": data.get("strength"),
        "prediction_error": data.get("prediction_error"),
        "score": data.get("score") or content.get("score"),
        "groundedness": data.get("groundedness") or content.get("groundedness"),
    }
    for key, value in numeric_candidates.items():
        parsed = _safe_float(value)
        if parsed is None:
            continue
        feats[f"value:{key}"] = max(0.0, min(1.0, parsed))

    return feats
