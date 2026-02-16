from __future__ import annotations

from typing import Any, Mapping


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _delta(after: Any, before: Any) -> float:
    return round(_safe_float(after) - _safe_float(before), 6)


def compute_trial_deltas(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, float]:
    b_ws = before.get("workspace") if isinstance(before.get("workspace"), Mapping) else {}
    a_ws = after.get("workspace") if isinstance(after.get("workspace"), Mapping) else {}
    b_rci = before.get("rci") if isinstance(before.get("rci"), Mapping) else {}
    a_rci = after.get("rci") if isinstance(after.get("rci"), Mapping) else {}

    return {
        "ignition_delta": _delta(a_ws.get("ignition_count"), b_ws.get("ignition_count")),
        "coherence_delta": _delta(after.get("coherence_ratio"), before.get("coherence_ratio")),
        "rci_delta": _delta(a_rci.get("rci"), b_rci.get("rci")),
        "agency_delta": _delta(after.get("agency"), before.get("agency")),
        "boundary_delta": _delta(after.get("boundary_stability"), before.get("boundary_stability")),
        "prediction_error_delta": _delta(after.get("world_prediction_error"), before.get("world_prediction_error")),
        "groundedness_delta": _delta(after.get("report_groundedness"), before.get("report_groundedness")),
        "trace_strength_delta": _delta(after.get("trace_strength"), before.get("trace_strength")),
    }


def composite_trial_score(deltas: Mapping[str, Any]) -> float:
    score = 0.0
    score += 0.25 * _safe_float(deltas.get("coherence_delta"))
    score += 0.25 * _safe_float(deltas.get("rci_delta"))
    score += 0.2 * _safe_float(deltas.get("trace_strength_delta"))
    score += 0.15 * _safe_float(deltas.get("agency_delta"))
    score += 0.15 * _safe_float(deltas.get("groundedness_delta"))
    return round(float(score), 6)
