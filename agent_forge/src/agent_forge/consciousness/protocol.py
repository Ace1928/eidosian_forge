from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

RAC_AP_PROTOCOL_VERSION = "rac_ap_protocol_v1_2026_02_19"


@dataclass(frozen=True)
class ProtocolValidationResult:
    valid: bool
    major_compatible: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    normalized: dict[str, Any]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp01(value: Any, default: float = 0.0) -> float:
    parsed = _safe_float(value, default)
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return float(parsed)


def protocol_major(version: str | None) -> Optional[int]:
    text = str(version or "").strip().lower()
    if not text:
        return None
    match = re.search(r"_v(\d+)", text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def default_nomological_expectations() -> list[dict[str, Any]]:
    return [
        {
            "name": "coherence_to_groundedness",
            "x": "coherence_ratio",
            "y": "report_groundedness",
            "relation": "positive",
            "threshold": 0.15,
        },
        {
            "name": "agency_to_ownership",
            "x": "agency",
            "y": "ownership_index",
            "relation": "positive",
            "threshold": 0.15,
        },
        {
            "name": "groundedness_to_perspective",
            "x": "report_groundedness",
            "y": "perspective_coherence_index",
            "relation": "positive",
            "threshold": 0.15,
        },
        {
            "name": "prediction_error_to_meta_confidence",
            "x": "prediction_error",
            "y": "meta_confidence",
            "relation": "negative",
            "threshold": 0.10,
        },
        {
            "name": "dream_vs_groundedness_control",
            "x": "dream_likeness_index",
            "y": "report_groundedness",
            "relation": "near_zero",
            "threshold": 0.35,
        },
    ]


def default_rac_ap_protocol() -> dict[str, Any]:
    return {
        "version": RAC_AP_PROTOCOL_VERSION,
        "construct": "robust_agentic_coherence_under_adversarial_perturbation",
        "pillars": [
            "global_availability_and_ignition_integrity",
            "predictive_self_correction",
            "goal_and_state_continuity",
            "boundary_integrity_under_attack",
            "causal_module_integration",
        ],
        "expectations": default_nomological_expectations(),
        "minimum_pairs": 6,
        "minimum_reports": {
            "bench_trials": 4,
            "benchmarks": 4,
        },
        "gates": {
            "reliability_min": 0.55,
            "convergent_min": 0.60,
            "discriminant_min": 0.55,
            "causal_min": 0.50,
            "rac_ap_index_min": 0.60,
            "security_required": False,
            "security_min": 0.60,
        },
    }


def validate_rac_ap_protocol(protocol: Mapping[str, Any]) -> ProtocolValidationResult:
    source = dict(protocol or {})
    normalized = default_rac_ap_protocol()
    errors: list[str] = []
    warnings: list[str] = []

    if source:
        normalized.update(
            {
                "version": str(source.get("version") or normalized.get("version")),
                "construct": str(source.get("construct") or normalized.get("construct")),
                "minimum_pairs": max(3, _safe_int(source.get("minimum_pairs"), int(normalized["minimum_pairs"]))),
            }
        )
        if isinstance(source.get("pillars"), list):
            normalized["pillars"] = [str(item) for item in source.get("pillars") if str(item)]
        if isinstance(source.get("expectations"), list):
            expected_rows: list[dict[str, Any]] = []
            for row in source.get("expectations"):
                if not isinstance(row, Mapping):
                    continue
                expected_rows.append(
                    {
                        "name": str(row.get("name") or ""),
                        "x": str(row.get("x") or ""),
                        "y": str(row.get("y") or ""),
                        "relation": str(row.get("relation") or "").lower(),
                        "threshold": _clamp01(row.get("threshold"), default=0.15),
                    }
                )
            if expected_rows:
                normalized["expectations"] = expected_rows
        if isinstance(source.get("minimum_reports"), Mapping):
            reports = dict(normalized["minimum_reports"])
            for key, default_val in reports.items():
                reports[key] = max(1, _safe_int(source.get("minimum_reports", {}).get(key), int(default_val)))
            normalized["minimum_reports"] = reports
        if isinstance(source.get("gates"), Mapping):
            gates = dict(normalized["gates"])
            for key, default_val in gates.items():
                incoming = source.get("gates", {}).get(key)
                if isinstance(default_val, bool):
                    gates[key] = bool(incoming) if incoming is not None else bool(default_val)
                else:
                    gates[key] = _clamp01(incoming, default=float(default_val))
            normalized["gates"] = gates

    expected_major = protocol_major(RAC_AP_PROTOCOL_VERSION)
    actual_major = protocol_major(str(normalized.get("version") or ""))
    major_compatible = expected_major is None or actual_major is None or int(expected_major) == int(actual_major)
    if not major_compatible:
        errors.append("Protocol major version is incompatible with runtime validator.")

    pillars = normalized.get("pillars")
    if not isinstance(pillars, list) or len(pillars) < 5:
        errors.append("Protocol must define at least five RAC-AP pillars.")

    expectations = normalized.get("expectations")
    if not isinstance(expectations, list) or len(expectations) < 3:
        errors.append("Protocol must define at least three nomological expectations.")
    else:
        allowed_rel = {"positive", "negative", "near_zero"}
        for idx, row in enumerate(expectations):
            if not isinstance(row, Mapping):
                errors.append(f"Expectation at index {idx} is not an object.")
                continue
            rel = str(row.get("relation") or "").lower()
            if rel not in allowed_rel:
                errors.append(f"Expectation '{row.get('name')}' has invalid relation '{rel}'.")
            if not str(row.get("x") or "") or not str(row.get("y") or ""):
                errors.append(f"Expectation '{row.get('name')}' requires non-empty x/y metrics.")

    gates = normalized.get("gates")
    required_gate_keys = {
        "reliability_min",
        "convergent_min",
        "discriminant_min",
        "causal_min",
        "rac_ap_index_min",
        "security_required",
        "security_min",
    }
    if not isinstance(gates, Mapping):
        errors.append("Protocol gates section is required.")
    else:
        missing = sorted(required_gate_keys - set(gates.keys()))
        if missing:
            errors.append(f"Protocol gates missing keys: {', '.join(missing)}")
        for key in required_gate_keys:
            if key == "security_required":
                continue
            value = gates.get(key)
            if value is None:
                continue
            parsed = _safe_float(value, -1.0)
            if parsed < 0.0 or parsed > 1.0:
                errors.append(f"Gate '{key}' must be in range [0,1].")

    if int(normalized.get("minimum_pairs") or 0) < 3:
        errors.append("minimum_pairs must be >= 3.")

    if not errors and not source:
        warnings.append("Validation used built-in protocol template.")

    return ProtocolValidationResult(
        valid=not errors,
        major_compatible=major_compatible,
        errors=tuple(errors),
        warnings=tuple(warnings),
        normalized=normalized,
    )


def read_protocol_file(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Protocol file must contain a top-level JSON object.")
    return payload


def write_protocol_file(path: str | Path, protocol: Mapping[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(protocol), indent=2, sort_keys=False), encoding="utf-8")
    return out_path


def default_preregistration(
    *,
    protocol: Mapping[str, Any],
    study_name: str,
    hypothesis: str,
    owner: str,
) -> dict[str, Any]:
    return {
        "prereg_id": f"prereg_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}",
        "created_at": _now_iso(),
        "protocol_version": str(protocol.get("version") or RAC_AP_PROTOCOL_VERSION),
        "construct": str(protocol.get("construct") or "robust_agentic_coherence_under_adversarial_perturbation"),
        "study_name": str(study_name or "rac_ap_study"),
        "owner": str(owner or "unknown"),
        "hypothesis": str(hypothesis or "").strip(),
        "expectations": list(protocol.get("expectations") or []),
        "falsification_gates": dict(protocol.get("gates") or {}),
        "minimum_pairs": int(protocol.get("minimum_pairs") or 6),
        "minimum_reports": dict(protocol.get("minimum_reports") or {}),
        "notes": [
            "Use deterministic seeds for all trial suites.",
            "Keep event-window capture digests for replay checks.",
            "Treat near_zero controls as hard discriminant checks.",
        ],
    }
