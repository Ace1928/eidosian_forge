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
    b_conn = before.get("connectivity") if isinstance(before.get("connectivity"), Mapping) else {}
    a_conn = after.get("connectivity") if isinstance(after.get("connectivity"), Mapping) else {}
    b_dir = before.get("directionality") if isinstance(before.get("directionality"), Mapping) else {}
    a_dir = after.get("directionality") if isinstance(after.get("directionality"), Mapping) else {}
    b_stab = before.get("self_stability") if isinstance(before.get("self_stability"), Mapping) else {}
    a_stab = after.get("self_stability") if isinstance(after.get("self_stability"), Mapping) else {}

    return {
        "ignition_delta": _delta(a_ws.get("ignition_count"), b_ws.get("ignition_count")),
        "coherence_delta": _delta(after.get("coherence_ratio"), before.get("coherence_ratio")),
        "rci_delta": _delta(a_rci.get("rci"), b_rci.get("rci")),
        "rci_v2_delta": _delta(a_rci.get("rci_v2"), b_rci.get("rci_v2")),
        "agency_delta": _delta(after.get("agency"), before.get("agency")),
        "boundary_delta": _delta(after.get("boundary_stability"), before.get("boundary_stability")),
        "prediction_error_delta": _delta(after.get("world_prediction_error"), before.get("world_prediction_error")),
        "groundedness_delta": _delta(after.get("report_groundedness"), before.get("report_groundedness")),
        "trace_strength_delta": _delta(after.get("trace_strength"), before.get("trace_strength")),
        "connectivity_density_delta": _delta(a_conn.get("density"), b_conn.get("density")),
        "workspace_centrality_delta": _delta(a_conn.get("workspace_centrality"), b_conn.get("workspace_centrality")),
        "directionality_delta": _delta(a_dir.get("mean_abs_asymmetry"), b_dir.get("mean_abs_asymmetry")),
        "self_stability_delta": _delta(a_stab.get("stability_score"), b_stab.get("stability_score")),
    }


def composite_trial_score(deltas: Mapping[str, Any]) -> float:
    score = 0.0
    score += 0.16 * _safe_float(deltas.get("coherence_delta"))
    score += 0.11 * _safe_float(deltas.get("rci_delta"))
    score += 0.11 * _safe_float(deltas.get("rci_v2_delta"))
    score += 0.11 * _safe_float(deltas.get("trace_strength_delta"))
    score += 0.11 * _safe_float(deltas.get("connectivity_density_delta"))
    score += 0.07 * _safe_float(deltas.get("workspace_centrality_delta"))
    score += 0.05 * _safe_float(deltas.get("directionality_delta"))
    score += 0.10 * _safe_float(deltas.get("agency_delta"))
    score += 0.10 * _safe_float(deltas.get("groundedness_delta"))
    score += 0.08 * _safe_float(deltas.get("self_stability_delta"))
    return round(float(score), 6)
