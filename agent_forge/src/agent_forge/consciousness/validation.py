from __future__ import annotations

import json
import math
import os
import statistics
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from agent_forge.core import events

from .bench.reporting import bench_report_root
from .protocol import (
    RAC_AP_PROTOCOL_VERSION,
    default_rac_ap_protocol,
    validate_rac_ap_protocol,
)


@dataclass(frozen=True)
class NomologicalExpectation:
    name: str
    x: str
    y: str
    relation: str
    threshold: float


def _expectations_from_protocol(protocol: Mapping[str, Any]) -> list[NomologicalExpectation]:
    rows = protocol.get("expectations") if isinstance(protocol.get("expectations"), Sequence) else []
    out: list[NomologicalExpectation] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        out.append(
            NomologicalExpectation(
                name=str(row.get("name") or f"expectation_{idx}"),
                x=str(row.get("x") or ""),
                y=str(row.get("y") or ""),
                relation=str(row.get("relation") or "positive"),
                threshold=float(_safe_float(row.get("threshold"), 0.15) or 0.15),
            )
        )
    return out


@dataclass
class ValidationResult:
    validation_id: str
    report_path: Optional[Path]
    report: dict[str, Any]


def _forge_root() -> Path:
    return Path(os.environ.get("EIDOS_FORGE_DIR", Path(__file__).resolve().parents[4])).resolve()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: Any, default: float = 0.0) -> float:
    parsed = _safe_float(value, default)
    if parsed is None:
        parsed = default
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return float(parsed)


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        payload["_path"] = str(path)
        payload["_mtime_ns"] = int(path.stat().st_mtime_ns)
        return payload
    return None


def _sorted_json_reports(paths: Iterable[Path], limit: int) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for path in paths:
        payload = _load_json(path)
        if payload is not None:
            loaded.append(payload)
    loaded.sort(key=lambda row: int(row.get("_mtime_ns") or 0), reverse=True)
    return loaded[: max(0, int(limit))]


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = min(len(xs), len(ys))
    if n < 3:
        return None
    x = list(float(v) for v in xs[:n])
    y = list(float(v) for v in ys[:n])
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((v - mx) ** 2 for v in x)
    syy = sum((v - my) ** 2 for v in y)
    if sxx <= 1e-12 or syy <= 1e-12:
        return None
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return float(sxy / math.sqrt(sxx * syy))


def _fisher_ci(corr: float, n: int) -> tuple[Optional[float], Optional[float]]:
    if n < 4:
        return (None, None)
    r = max(-0.999999, min(0.999999, float(corr)))
    z = 0.5 * math.log((1.0 + r) / (1.0 - r))
    se = 1.0 / math.sqrt(max(n - 3, 1))
    z_lo = z - (1.96 * se)
    z_hi = z + (1.96 * se)
    lo = math.tanh(z_lo)
    hi = math.tanh(z_hi)
    return (round(float(lo), 6), round(float(hi), 6))


def _metric_from_vector(row: Mapping[str, Any], key: str) -> Optional[float]:
    return _safe_float(row.get(key), None)


def _collect_pairs(vectors: Sequence[Mapping[str, Any]], x: str, y: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in vectors:
        xv = _metric_from_vector(row, x)
        yv = _metric_from_vector(row, y)
        if xv is None or yv is None:
            continue
        xs.append(float(xv))
        ys.append(float(yv))
    return xs, ys


def _relation_pass(relation: str, corr: float, threshold: float) -> bool:
    rel = str(relation or "").strip().lower()
    if rel == "positive":
        return corr >= threshold
    if rel == "negative":
        return corr <= -threshold
    if rel == "near_zero":
        return abs(corr) <= threshold
    return False


def _relation_score(relation: str, corr: float, threshold: float) -> float:
    rel = str(relation or "").strip().lower()
    if rel == "positive":
        if threshold <= 1e-9:
            return _clamp01(corr)
        return _clamp01(corr / threshold)
    if rel == "negative":
        if threshold <= 1e-9:
            return _clamp01(-corr)
        return _clamp01((-corr) / threshold)
    if rel == "near_zero":
        if threshold <= 1e-9:
            return 0.0
        return _clamp01(1.0 - (abs(corr) / threshold))
    return 0.0


class ConsciousnessConstructValidator:
    def __init__(self, state_dir: str | Path) -> None:
        self.state_dir = Path(state_dir)

    def _validation_dir(self) -> Path:
        default = _forge_root() / "reports" / "consciousness_validation"
        path = Path(os.environ.get("EIDOS_CONSCIOUSNESS_VALIDATION_DIR", str(default))).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _benchmark_dir(self) -> Path:
        default = _forge_root() / "reports" / "consciousness_benchmarks"
        path = Path(os.environ.get("EIDOS_CONSCIOUSNESS_BENCHMARK_DIR", str(default))).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _integrated_dir(self) -> Path:
        default = _forge_root() / "reports" / "consciousness_integrated_benchmarks"
        path = Path(os.environ.get("EIDOS_CONSCIOUSNESS_INTEGRATED_BENCH_DIR", str(default))).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _legacy_trial_dir(self) -> Path:
        default = _forge_root() / "reports" / "consciousness_trials"
        path = Path(os.environ.get("EIDOS_CONSCIOUSNESS_TRIAL_DIR", str(default))).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load_bench_trial_reports(self, limit: int) -> list[dict[str, Any]]:
        root = bench_report_root(self.state_dir)
        paths = sorted(root.glob("*/report.json"))
        filtered = [p for p in paths if not p.parent.name.startswith("red_team_")]
        return _sorted_json_reports(filtered, limit)

    def _load_benchmark_reports(self, limit: int) -> list[dict[str, Any]]:
        return _sorted_json_reports(self._benchmark_dir().glob("benchmark_*.json"), limit)

    def _load_integrated_reports(self, limit: int) -> list[dict[str, Any]]:
        return _sorted_json_reports(self._integrated_dir().glob("integrated_*.json"), limit)

    def _load_legacy_trial_reports(self, limit: int) -> list[dict[str, Any]]:
        return _sorted_json_reports(self._legacy_trial_dir().glob("trial_*.json"), limit)

    def _load_red_team_reports(self, limit: int) -> list[dict[str, Any]]:
        root = bench_report_root(self.state_dir)
        return _sorted_json_reports(root.glob("red_team_*/report.json"), limit)

    def _vector_from_bench_trial(self, report: Mapping[str, Any]) -> dict[str, Any]:
        after = report.get("after") if isinstance(report.get("after"), Mapping) else {}
        phenom = after.get("phenomenology") if isinstance(after.get("phenomenology"), Mapping) else {}
        return {
            "source": "bench_trial",
            "coherence_ratio": _safe_float(after.get("coherence_ratio"), None),
            "trace_strength": _safe_float(after.get("trace_strength"), None),
            "agency": _safe_float(after.get("agency"), None),
            "boundary_stability": _safe_float(after.get("boundary_stability"), None),
            "prediction_error": _safe_float(after.get("world_prediction_error"), None),
            "meta_confidence": _safe_float(after.get("meta_confidence"), None),
            "report_groundedness": _safe_float(after.get("report_groundedness"), None),
            "unity_index": _safe_float(phenom.get("unity_index"), None),
            "continuity_index": _safe_float(phenom.get("continuity_index"), None),
            "ownership_index": _safe_float(phenom.get("ownership_index"), None),
            "perspective_coherence_index": _safe_float(phenom.get("perspective_coherence_index"), None),
            "dream_likeness_index": _safe_float(phenom.get("dream_likeness_index"), None),
            "trial_composite": _safe_float(report.get("composite_score"), None),
        }

    def _vector_from_legacy_trial(self, report: Mapping[str, Any]) -> dict[str, Any]:
        after = report.get("after") if isinstance(report.get("after"), Mapping) else {}
        coherence = after.get("coherence") if isinstance(after.get("coherence"), Mapping) else {}
        phenom = after.get("phenomenology") if isinstance(after.get("phenomenology"), Mapping) else {}
        return {
            "source": "legacy_trial",
            "coherence_ratio": _safe_float(coherence.get("coherence_ratio"), None),
            "trace_strength": _safe_float(after.get("trace_strength"), None),
            "agency": _safe_float(after.get("agency"), None),
            "boundary_stability": _safe_float(after.get("boundary_stability"), None),
            "prediction_error": _safe_float(after.get("world_prediction_error"), None),
            "meta_confidence": _safe_float(after.get("meta_confidence"), None),
            "report_groundedness": _safe_float(after.get("report_groundedness"), None),
            "unity_index": _safe_float(after.get("unity_index"), _safe_float(phenom.get("unity_index"), None)),
            "continuity_index": _safe_float(
                after.get("continuity_index"), _safe_float(phenom.get("continuity_index"), None)
            ),
            "ownership_index": _safe_float(
                after.get("ownership_index"), _safe_float(phenom.get("ownership_index"), None)
            ),
            "perspective_coherence_index": _safe_float(
                after.get("perspective_coherence_index"),
                _safe_float(phenom.get("perspective_coherence_index"), None),
            ),
            "dream_likeness_index": _safe_float(
                after.get("dream_likeness_index"), _safe_float(phenom.get("dream_likeness_index"), None)
            ),
            "trial_composite": _safe_float(((report.get("delta") or {}).get("coherence_delta")), None),
        }

    def _vector_from_benchmark(self, report: Mapping[str, Any]) -> dict[str, Any]:
        cap = report.get("capability") if isinstance(report.get("capability"), Mapping) else {}
        scores = report.get("scores") if isinstance(report.get("scores"), Mapping) else {}
        return {
            "source": "benchmark",
            "coherence_ratio": _safe_float(cap.get("coherence_ratio"), None),
            "trace_strength": _safe_float(cap.get("trace_strength"), None),
            "agency": _safe_float(cap.get("agency"), None),
            "boundary_stability": _safe_float(cap.get("boundary_stability"), None),
            "prediction_error": _safe_float(cap.get("prediction_error"), None),
            "meta_confidence": _safe_float(cap.get("meta_confidence"), None),
            "report_groundedness": _safe_float(cap.get("report_groundedness"), None),
            "unity_index": _safe_float(cap.get("phenom_unity_index"), None),
            "continuity_index": _safe_float(cap.get("phenom_continuity_index"), None),
            "ownership_index": _safe_float(cap.get("phenom_ownership_index"), None),
            "perspective_coherence_index": _safe_float(cap.get("phenom_perspective_coherence_index"), None),
            "dream_likeness_index": _safe_float(cap.get("phenom_dream_likeness_index"), None),
            "trial_composite": _safe_float(scores.get("composite"), None),
        }

    def _vector_from_integrated(self, report: Mapping[str, Any]) -> dict[str, Any]:
        scores = report.get("scores") if isinstance(report.get("scores"), Mapping) else {}
        red = report.get("red_team") if isinstance(report.get("red_team"), Mapping) else {}
        return {
            "source": "integrated",
            "coherence_ratio": _safe_float(scores.get("core_score"), None),
            "trace_strength": _safe_float(scores.get("trial_score"), None),
            "agency": _safe_float(scores.get("integrated"), None),
            "boundary_stability": _safe_float(red.get("mean_robustness"), None),
            "prediction_error": None,
            "meta_confidence": None,
            "report_groundedness": _safe_float(scores.get("llm_score"), None),
            "unity_index": None,
            "continuity_index": None,
            "ownership_index": None,
            "perspective_coherence_index": None,
            "dream_likeness_index": None,
            "trial_composite": _safe_float(scores.get("integrated"), None),
        }

    def _reliability_summary(self, vectors: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        def _cv_for(metric: str) -> dict[str, Any]:
            vals = [float(v) for v in (_metric_from_vector(row, metric) for row in vectors) if v is not None]
            if len(vals) < 2:
                return {"count": len(vals), "mean": None, "stdev": None, "cv": None, "score": 0.0}
            mean = statistics.fmean(vals)
            stdev = statistics.pstdev(vals)
            cv = None
            if abs(mean) > 1e-9:
                cv = abs(stdev / mean)
            score = _clamp01(1.0 - ((cv or 1.0) / 0.5), default=0.0)
            return {
                "count": len(vals),
                "mean": round(float(mean), 6),
                "stdev": round(float(stdev), 6),
                "cv": round(float(cv), 6) if cv is not None else None,
                "score": round(float(score), 6),
            }

        tracked = {
            "coherence_ratio": _cv_for("coherence_ratio"),
            "agency": _cv_for("agency"),
            "report_groundedness": _cv_for("report_groundedness"),
            "trial_composite": _cv_for("trial_composite"),
        }
        scores = [float(v.get("score") or 0.0) for v in tracked.values()]
        return {
            "metrics": tracked,
            "score": round(float(sum(scores) / max(len(scores), 1)), 6),
        }

    def _expectation_checks(
        self,
        vectors: Sequence[Mapping[str, Any]],
        *,
        expectations: Sequence[NomologicalExpectation],
        min_pairs: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for exp in expectations:
            xs, ys = _collect_pairs(vectors, exp.x, exp.y)
            n = min(len(xs), len(ys))
            corr = _pearson(xs, ys)
            ci_low, ci_high = (None, None)
            passed = False
            quality = 0.0
            if corr is not None:
                ci_low, ci_high = _fisher_ci(corr, n)
                if n >= min_pairs:
                    passed = _relation_pass(exp.relation, corr, exp.threshold)
                    quality = _relation_score(exp.relation, corr, exp.threshold)
            rows.append(
                {
                    "name": exp.name,
                    "x": exp.x,
                    "y": exp.y,
                    "relation": exp.relation,
                    "threshold": float(exp.threshold),
                    "pairs": int(n),
                    "corr": round(float(corr), 6) if corr is not None else None,
                    "corr_ci95": [ci_low, ci_high],
                    "pass": bool(passed),
                    "quality": round(float(quality), 6),
                }
            )
        return rows

    def _security_summary(self, red_team_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not red_team_reports:
            return {
                "available": False,
                "score": 0.0,
                "pass_ratio": None,
                "mean_robustness": None,
                "attack_success_rate": None,
                "runs": 0,
            }
        pass_ratios: list[float] = []
        robustness: list[float] = []
        attack_success_rates: list[float] = []
        for rep in red_team_reports:
            pass_ratio = _safe_float(rep.get("pass_ratio"), None)
            robust = _safe_float(rep.get("mean_robustness"), None)
            fail_count = _safe_float(rep.get("fail_count"), 0.0) or 0.0
            scenario_count = _safe_float(rep.get("scenario_count"), 0.0) or 0.0
            attack_success = (fail_count / scenario_count) if scenario_count > 0 else None
            if pass_ratio is not None:
                pass_ratios.append(float(pass_ratio))
            if robust is not None:
                robustness.append(float(robust))
            if attack_success is not None:
                attack_success_rates.append(float(attack_success))

        mean_pass_ratio = sum(pass_ratios) / max(len(pass_ratios), 1) if pass_ratios else None
        mean_robustness = sum(robustness) / max(len(robustness), 1) if robustness else None
        mean_attack_success = (
            sum(attack_success_rates) / max(len(attack_success_rates), 1) if attack_success_rates else None
        )
        pass_score = _clamp01(mean_pass_ratio if mean_pass_ratio is not None else 0.0)
        robustness_score = _clamp01(mean_robustness if mean_robustness is not None else 0.0)
        score = (0.6 * pass_score) + (0.4 * robustness_score)
        return {
            "available": True,
            "runs": len(red_team_reports),
            "pass_ratio": round(float(mean_pass_ratio), 6) if mean_pass_ratio is not None else None,
            "mean_robustness": round(float(mean_robustness), 6) if mean_robustness is not None else None,
            "attack_success_rate": (round(float(mean_attack_success), 6) if mean_attack_success is not None else None),
            "score": round(float(score), 6),
        }

    def _intervention_summary(self, bench_trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not bench_trials:
            return {
                "available": False,
                "score": 0.0,
                "sample_count": 0,
                "intervention_count": 0,
                "expected_signature_pass_ratio": None,
                "by_intervention": [],
            }

        tracked_metrics = (
            "coherence_delta",
            "ignition_delta",
            "trace_strength_delta",
            "agency_delta",
            "prediction_error_delta",
            "groundedness_delta",
        )
        grouped: dict[str, list[dict[str, Any]]] = {}
        expected_passes = 0
        expected_total = 0
        for report in bench_trials:
            deltas = report.get("deltas") if isinstance(report.get("deltas"), Mapping) else {}
            perturbations = report.get("perturbations") if isinstance(report.get("perturbations"), Sequence) else []
            recipe_eval = (
                report.get("recipe_expectations") if isinstance(report.get("recipe_expectations"), Mapping) else {}
            )
            if bool(recipe_eval.get("defined")):
                expected_total += 1
                if bool(recipe_eval.get("pass")):
                    expected_passes += 1
            if not perturbations:
                continue
            for perturb in perturbations:
                if not isinstance(perturb, Mapping):
                    continue
                key = f"{str(perturb.get('kind') or 'unknown')}:" f"{str(perturb.get('target') or 'unknown')}"
                row = {
                    "trial_id": str(report.get("trial_id") or ""),
                    "magnitude": float(_safe_float(perturb.get("magnitude"), 0.0) or 0.0),
                    "duration_s": float(_safe_float(perturb.get("duration_s"), 0.0) or 0.0),
                    "deltas": {
                        metric: float(_safe_float(deltas.get(metric), 0.0) or 0.0) for metric in tracked_metrics
                    },
                }
                grouped.setdefault(key, []).append(row)

        by_intervention: list[dict[str, Any]] = []
        weighted_scores: list[float] = []
        for key, rows in sorted(grouped.items()):
            n = len(rows)
            delta_rows = [row.get("deltas") if isinstance(row.get("deltas"), Mapping) else {} for row in rows]
            metric_stats: dict[str, dict[str, Any]] = {}
            effect_strength = 0.0
            consistency_vals: list[float] = []
            available_metric_count = 0
            for metric in tracked_metrics:
                vals = [float(_safe_float(d.get(metric), 0.0) or 0.0) for d in delta_rows]
                if not vals:
                    metric_stats[metric] = {"mean": None, "abs_mean": None, "sign_consistency": None}
                    continue
                mean_val = sum(vals) / len(vals)
                abs_mean = sum(abs(v) for v in vals) / len(vals)
                sign_anchor = 1.0 if mean_val >= 0 else -1.0
                sign_consistency = sum(
                    1.0 for v in vals if (v == 0.0 or (1.0 if v >= 0 else -1.0) == sign_anchor)
                ) / len(vals)
                metric_stats[metric] = {
                    "mean": round(float(mean_val), 6),
                    "abs_mean": round(float(abs_mean), 6),
                    "sign_consistency": round(float(sign_consistency), 6),
                }
                effect_strength += abs_mean
                consistency_vals.append(sign_consistency)
                available_metric_count += 1
            effect_strength = effect_strength / float(available_metric_count) if available_metric_count > 0 else 0.0
            consistency = sum(consistency_vals) / float(len(consistency_vals)) if consistency_vals else 0.0
            sample_weight = min(1.0, float(n) / 4.0)
            score = _clamp01(((effect_strength / 0.15) * 0.6) + (consistency * 0.4)) * sample_weight
            weighted_scores.append(score)
            by_intervention.append(
                {
                    "intervention": key,
                    "sample_count": n,
                    "effect_strength": round(float(effect_strength), 6),
                    "consistency": round(float(consistency), 6),
                    "score": round(float(score), 6),
                    "metric_deltas": metric_stats,
                }
            )

        expected_signature_pass_ratio = (
            round(float(expected_passes) / float(expected_total), 6) if expected_total > 0 else None
        )
        mean_score = sum(weighted_scores) / float(len(weighted_scores)) if weighted_scores else 0.0
        if expected_signature_pass_ratio is not None:
            mean_score = (0.7 * mean_score) + (0.3 * float(expected_signature_pass_ratio))
        mean_score = _clamp01(mean_score)
        return {
            "available": bool(grouped),
            "score": round(float(mean_score), 6),
            "sample_count": int(sum(len(v) for v in grouped.values())),
            "intervention_count": len(grouped),
            "expected_signature_pass_ratio": expected_signature_pass_ratio,
            "by_intervention": sorted(by_intervention, key=lambda row: float(row.get("score") or 0.0), reverse=True),
        }

    def run(
        self,
        *,
        limit: int = 64,
        persist: bool = True,
        min_pairs: int | None = None,
        protocol: Mapping[str, Any] | None = None,
        security_required: bool | None = None,
    ) -> ValidationResult:
        protocol_input = dict(protocol or default_rac_ap_protocol())
        protocol_check = validate_rac_ap_protocol(protocol_input)
        protocol_data = protocol_check.normalized
        if isinstance(security_required, bool):
            gates = protocol_data.get("gates")
            if isinstance(gates, Mapping):
                mutable_gates = dict(gates)
                mutable_gates["security_required"] = bool(security_required)
                protocol_data["gates"] = mutable_gates
        protocol_expectations = _expectations_from_protocol(protocol_data)

        min_pairs = int(min_pairs or protocol_data.get("minimum_pairs") or 6)
        min_pairs = max(3, min_pairs)

        bench_trials = self._load_bench_trial_reports(limit)
        legacy_trials = self._load_legacy_trial_reports(limit)
        benchmarks = self._load_benchmark_reports(limit)
        integrated = self._load_integrated_reports(limit)
        red_team = self._load_red_team_reports(limit)

        vectors: list[dict[str, Any]] = []
        vectors.extend(self._vector_from_bench_trial(row) for row in bench_trials)
        vectors.extend(self._vector_from_legacy_trial(row) for row in legacy_trials)
        vectors.extend(self._vector_from_benchmark(row) for row in benchmarks)
        vectors.extend(self._vector_from_integrated(row) for row in integrated)

        reliability = self._reliability_summary(vectors)
        checks = self._expectation_checks(
            vectors,
            expectations=protocol_expectations,
            min_pairs=min_pairs,
        )
        convergent_checks = [row for row in checks if row.get("relation") in {"positive", "negative"}]
        discriminant_checks = [row for row in checks if row.get("relation") == "near_zero"]
        intervention = self._intervention_summary(bench_trials)

        def _mean_quality(rows: Sequence[Mapping[str, Any]]) -> float:
            if not rows:
                return 0.0
            return float(sum(float(r.get("quality") or 0.0) for r in rows) / len(rows))

        convergent_score = _mean_quality(convergent_checks)
        discriminant_score = _mean_quality(discriminant_checks)
        security = self._security_summary(red_team)

        gates_cfg = protocol_data.get("gates") if isinstance(protocol_data.get("gates"), Mapping) else {}
        min_reports_cfg = (
            protocol_data.get("minimum_reports") if isinstance(protocol_data.get("minimum_reports"), Mapping) else {}
        )
        require_security = bool(gates_cfg.get("security_required", False))
        security_gate = (
            (
                bool(security.get("available"))
                and float(security.get("score") or 0.0) >= float(gates_cfg.get("security_min", 0.6))
            )
            if require_security
            else True
        )

        weighted_components: list[tuple[str, float, float]] = [
            ("reliability", 0.25, float(reliability.get("score") or 0.0)),
            ("convergent", 0.25, float(convergent_score)),
            ("discriminant", 0.15, float(discriminant_score)),
            ("interventional", 0.15, float(intervention.get("score") or 0.0)),
        ]
        if require_security or bool(security.get("available")):
            weighted_components.append(("security", 0.20, float(security.get("score") or 0.0)))
        total_weight = sum(weight for _, weight, _ in weighted_components) or 1.0
        overall = sum(weight * score for _, weight, score in weighted_components) / total_weight

        gates = {
            "protocol_valid": bool(protocol_check.valid),
            "protocol_major_compatible": bool(protocol_check.major_compatible),
            "reliability_min": float(reliability.get("score") or 0.0) >= float(gates_cfg.get("reliability_min", 0.55)),
            "convergent_min": convergent_score >= float(gates_cfg.get("convergent_min", 0.6)),
            "discriminant_min": discriminant_score >= float(gates_cfg.get("discriminant_min", 0.55)),
            "causal_min": float(intervention.get("score") or 0.0) >= float(gates_cfg.get("causal_min", 0.5)),
            "rac_ap_index_min": overall >= float(gates_cfg.get("rac_ap_index_min", 0.6)),
            "security_min": security_gate,
            "minimum_data": len(vectors) >= min_pairs,
            "minimum_reports": (
                len(bench_trials) >= int(min_reports_cfg.get("bench_trials", 1))
                and len(benchmarks) >= int(min_reports_cfg.get("benchmarks", 1))
            ),
        }

        failed_expectations = [row["name"] for row in checks if not bool(row.get("pass"))]
        recommendations: list[str] = []
        if not gates["protocol_valid"]:
            recommendations.append(
                "Protocol schema is invalid; fix protocol definition before interpreting RAC-AP scores."
            )
        if not gates["protocol_major_compatible"]:
            recommendations.append("Protocol major version mismatch; re-export a compatible protocol version.")
        if not gates["minimum_data"]:
            recommendations.append("Increase benchmark/trial sample count before interpreting RAC-AP scores.")
        if not gates["minimum_reports"]:
            recommendations.append("Collect minimum bench trial/benchmark counts required by protocol before scoring.")
        if not gates["convergent_min"]:
            recommendations.append(
                "Strengthen positive/negative expected coupling under perturbation and ablation tests."
            )
        if not gates["discriminant_min"]:
            recommendations.append("Revisit proxy metrics; discriminant controls are too entangled.")
        if not gates["causal_min"]:
            recommendations.append(
                "Intervention signatures are weak or inconsistent; improve perturbation observability and do-style controls."
            )
        if require_security and not gates["security_min"]:
            recommendations.append("Improve boundary integrity against red-team attack scenarios.")

        validation_id = f"validation_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}"
        report = {
            "validation_id": validation_id,
            "timestamp": _now_iso(),
            "state_dir": str(self.state_dir),
            "protocol": protocol_data,
            "protocol_compatibility": {
                "runtime_version": RAC_AP_PROTOCOL_VERSION,
                "provided_version": str(protocol_data.get("version") or ""),
                "valid": bool(protocol_check.valid),
                "major_compatible": bool(protocol_check.major_compatible),
                "errors": list(protocol_check.errors),
                "warnings": list(protocol_check.warnings),
            },
            "sample_sizes": {
                "vectors": len(vectors),
                "bench_trials": len(bench_trials),
                "legacy_trials": len(legacy_trials),
                "benchmarks": len(benchmarks),
                "integrated": len(integrated),
                "red_team": len(red_team),
            },
            "reliability": reliability,
            "nomological_checks": checks,
            "convergent_validity": {
                "score": round(float(convergent_score), 6),
                "checks": len(convergent_checks),
                "pass_count": sum(1 for row in convergent_checks if bool(row.get("pass"))),
            },
            "discriminant_validity": {
                "score": round(float(discriminant_score), 6),
                "checks": len(discriminant_checks),
                "pass_count": sum(1 for row in discriminant_checks if bool(row.get("pass"))),
            },
            "interventional_validity": intervention,
            "security_boundary": security,
            "falsification": {
                "failed_expectation_count": len(failed_expectations),
                "failed_expectations": failed_expectations,
            },
            "scores": {
                "rac_ap_index": round(float(overall), 6),
                "weighted_components": [
                    {
                        "name": name,
                        "weight": round(float(weight), 6),
                        "score": round(float(score), 6),
                    }
                    for name, weight, score in weighted_components
                ],
            },
            "gates": gates,
            "pass": bool(all(gates.values())),
            "recommendations": recommendations,
        }

        events.append(
            self.state_dir,
            "validation.rac_ap_result",
            {
                "validation_id": validation_id,
                "rac_ap_index": report["scores"]["rac_ap_index"],
                "pass": bool(report.get("pass")),
                "gates": gates,
                "sample_sizes": report.get("sample_sizes"),
            },
            tags=["consciousness", "validation", "rac_ap"],
        )

        path: Optional[Path] = None
        if persist:
            path = self._validation_dir() / f"{validation_id}.json"
            path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            report["report_path"] = str(path)

        return ValidationResult(validation_id=validation_id, report_path=path, report=report)

    def latest_validation(self) -> Optional[dict[str, Any]]:
        files = sorted(self._validation_dir().glob("validation_*.json"))
        if not files:
            return None
        latest = max(files, key=lambda p: p.stat().st_mtime_ns)
        return _load_json(latest)

    def _validation_reports(self, limit: int = 2) -> list[dict[str, Any]]:
        return _sorted_json_reports(self._validation_dir().glob("validation_*.json"), limit)

    def _protocol_threshold_map(self, protocol: Mapping[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}

        minimum_pairs = _safe_float(protocol.get("minimum_pairs"), None)
        if minimum_pairs is not None:
            out["minimum_pairs"] = float(minimum_pairs)

        gates = protocol.get("gates")
        if isinstance(gates, Mapping):
            for key, value in gates.items():
                if isinstance(value, bool):
                    continue
                parsed = _safe_float(value, None)
                if parsed is not None:
                    out[f"gates.{key}"] = float(parsed)

        minimum_reports = protocol.get("minimum_reports")
        if isinstance(minimum_reports, Mapping):
            for key, value in minimum_reports.items():
                parsed = _safe_float(value, None)
                if parsed is not None:
                    out[f"minimum_reports.{key}"] = float(parsed)

        expectations = protocol.get("expectations")
        if isinstance(expectations, Sequence):
            for idx, row in enumerate(expectations):
                if not isinstance(row, Mapping):
                    continue
                name = str(row.get("name") or f"expectation_{idx}")
                threshold = _safe_float(row.get("threshold"), None)
                if threshold is None:
                    continue
                out[f"expectations.{name}.threshold"] = float(threshold)

        return out

    def protocol_drift_review(
        self,
        *,
        threshold: float = 0.05,
        current_protocol: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        reports = self._validation_reports(limit=2)
        if not reports:
            return {"error": "No validation reports found"}

        current_report = reports[0]
        baseline_report = reports[1] if len(reports) > 1 else None
        protocol_now = dict(current_protocol) if current_protocol is not None else (
            current_report.get("protocol") if isinstance(current_report.get("protocol"), Mapping) else {}
        )
        protocol_before = (
            baseline_report.get("protocol")
            if baseline_report and isinstance(baseline_report.get("protocol"), Mapping)
            else {}
        )

        now_map = self._protocol_threshold_map(protocol_now)
        before_map = self._protocol_threshold_map(protocol_before)
        keys = sorted(set(now_map.keys()) | set(before_map.keys()))

        rows: list[dict[str, Any]] = []
        flagged = 0
        for key in keys:
            new_val = now_map.get(key)
            old_val = before_map.get(key)
            status = "unchanged"
            delta = None
            drift_flag = False
            if old_val is None and new_val is not None:
                status = "added"
                drift_flag = True
            elif new_val is None and old_val is not None:
                status = "removed"
                drift_flag = True
            elif old_val is not None and new_val is not None:
                delta = float(new_val - old_val)
                if abs(delta) >= max(0.0, float(threshold)):
                    status = "changed"
                    drift_flag = True
                elif abs(delta) > 0:
                    status = "changed_minor"
            if drift_flag:
                flagged += 1
            rows.append(
                {
                    "key": key,
                    "status": status,
                    "baseline": old_val,
                    "current": new_val,
                    "delta": round(delta, 6) if delta is not None else None,
                    "drift_flag": drift_flag,
                }
            )

        return {
            "timestamp": _now_iso(),
            "threshold": float(threshold),
            "comparison": {
                "current_validation_id": current_report.get("validation_id"),
                "baseline_validation_id": baseline_report.get("validation_id") if baseline_report else None,
                "reports_compared": 2 if baseline_report else 1,
            },
            "summary": {
                "total_keys": len(keys),
                "flagged_count": flagged,
                "has_drift": bool(flagged > 0),
            },
            "changes": rows,
        }
