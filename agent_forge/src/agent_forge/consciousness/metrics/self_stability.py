from __future__ import annotations

from typing import Any, Mapping, Sequence


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stats(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return float(mean), float(var)


def _volatility(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    diffs = [abs(b - a) for a, b in zip(values, values[1:])]
    return float(sum(diffs) / len(diffs)) if diffs else 0.0


def self_stability(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    agency_vals: list[float] = []
    boundary_vals: list[float] = []

    for evt in events:
        etype = str(evt.get("type") or "")
        data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
        if etype == "metrics.sample":
            key = str(data.get("key") or "")
            val = _safe_float(data.get("value"))
            if val is None:
                continue
            if key == "consciousness.agency":
                agency_vals.append(max(0.0, min(1.0, val)))
            if key == "consciousness.boundary_stability":
                boundary_vals.append(max(0.0, min(1.0, val)))
        elif etype == "self.agency_estimate":
            val = _safe_float(data.get("agency_confidence"))
            if val is not None:
                agency_vals.append(max(0.0, min(1.0, val)))
        elif etype == "self.boundary_estimate":
            val = _safe_float(data.get("boundary_stability"))
            if val is not None:
                boundary_vals.append(max(0.0, min(1.0, val)))

    agency_mean, agency_var = _stats(agency_vals)
    boundary_mean, boundary_var = _stats(boundary_vals)
    # Max variance for [0,1] bounded signals is 0.25; normalize to [0,1].
    agency_var_n = min(1.0, agency_var / 0.25) if agency_vals else 1.0
    boundary_var_n = min(1.0, boundary_var / 0.25) if boundary_vals else 1.0
    stability_score = 1.0 - ((agency_var_n + boundary_var_n) / 2.0)
    stability_score = max(0.0, min(1.0, stability_score))

    return {
        "sample_count": int(max(len(agency_vals), len(boundary_vals))),
        "agency_mean": round(agency_mean, 6),
        "agency_variance": round(agency_var, 6),
        "agency_volatility": round(_volatility(agency_vals), 6),
        "boundary_mean": round(boundary_mean, 6),
        "boundary_variance": round(boundary_var, 6),
        "boundary_volatility": round(_volatility(boundary_vals), 6),
        "stability_score": round(stability_score, 6),
    }
