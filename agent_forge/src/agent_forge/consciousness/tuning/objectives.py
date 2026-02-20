from __future__ import annotations

from typing import Any, Mapping


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def objectives_from_trial_report(report: Mapping[str, Any]) -> dict[str, float]:
    deltas = report.get("deltas") if isinstance(report.get("deltas"), Mapping) else {}
    degrade_ratio = _safe_float(report.get("degraded_mode_ratio"), 0.0)
    trace_violations = _safe_float(report.get("ignitions_without_trace"), 0.0)
    module_errors = _safe_float(report.get("module_error_count"), 0.0)

    return {
        "coherence": _safe_float(deltas.get("coherence_delta"), 0.0),
        "trace_strength": _safe_float(deltas.get("trace_strength_delta"), 0.0),
        "ownership": _safe_float(deltas.get("ownership_delta"), 0.0),
        "groundedness": _safe_float(deltas.get("groundedness_delta"), 0.0),
        "continuity": _safe_float(deltas.get("continuity_delta"), 0.0),
        "self_stability": _safe_float(deltas.get("self_stability_delta"), 0.0),
        "anti_degraded": -degrade_ratio,
        "anti_trace_violation": -trace_violations,
        "anti_module_error": -module_errors,
    }
