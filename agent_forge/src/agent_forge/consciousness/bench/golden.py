from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class GoldenCheck:
    metric: str
    op: str
    threshold: float


DEFAULT_GOLDENS: dict[str, list[GoldenCheck]] = {
    "no_competition": [
        GoldenCheck("trace_strength_delta_vs_full", "<=", -0.05),
        GoldenCheck("coherence_ratio_delta_vs_full", "<=", -0.02),
    ],
    "no_working_set": [
        GoldenCheck("connectivity_density_delta_vs_full", "<=", -0.01),
    ],
    "no_self_model_ext": [
        GoldenCheck("self_stability_delta_vs_full", "<=", -0.01),
    ],
    "no_meta": [
        GoldenCheck("composite_delta_vs_full", "<=", -0.01),
    ],
    "no_world_model": [
        GoldenCheck("rci_v2_delta_vs_full", "<=", -0.01),
    ],
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _compare(value: float, op: str, threshold: float) -> bool:
    if op == "<=":
        return value <= threshold
    if op == ">=":
        return value >= threshold
    if op == "<":
        return value < threshold
    if op == ">":
        return value > threshold
    if op == "==":
        return value == threshold
    return False


def evaluate_variant_golden(
    variant_name: str,
    comparison: Mapping[str, Any],
    *,
    golden_map: Mapping[str, list[GoldenCheck]] | None = None,
) -> dict[str, Any]:
    rules = dict(golden_map or DEFAULT_GOLDENS)
    checks = rules.get(str(variant_name), [])
    out_checks: list[dict[str, Any]] = []
    ok = True
    for rule in checks:
        value = _safe_float(comparison.get(rule.metric), default=0.0)
        passed = _compare(value, rule.op, float(rule.threshold))
        out_checks.append(
            {
                "metric": rule.metric,
                "op": rule.op,
                "threshold": float(rule.threshold),
                "value": round(value, 6),
                "pass": bool(passed),
            }
        )
        if not passed:
            ok = False
    return {
        "variant": str(variant_name),
        "defined": bool(checks),
        "checks": out_checks,
        "pass": bool(ok),
    }
