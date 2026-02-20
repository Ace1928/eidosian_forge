from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from agent_forge.core import events

from .reporting import bench_report_root, write_json, write_summary
from .trials import ConsciousnessBenchRunner, TrialSpec


@dataclass(frozen=True)
class RedTeamScenario:
    name: str
    description: str
    task: str
    perturbations: tuple[dict[str, Any], ...]
    disable_modules: tuple[str, ...] = ()
    warmup_beats: int = 1
    baseline_seconds: float = 1.2
    perturb_seconds: float = 1.4
    recovery_seconds: float = 1.2
    beat_seconds: float = 0.2
    checks: dict[str, Any] = field(default_factory=dict)


@dataclass
class RedTeamResult:
    run_id: str
    report_path: Optional[Path]
    report: dict[str, Any]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _default_checks() -> dict[str, Any]:
    return {
        "max_module_errors": 0,
        "max_degraded_mode_ratio": 0.75,
        "max_winner_count": 240,
        "max_ignitions_without_trace": 0,
        "min_report_groundedness": 0.15,
        "min_trace_strength": 0.02,
        "require_recipe_expectations": True,
    }


def default_red_team_scenarios() -> list[RedTeamScenario]:
    return [
        RedTeamScenario(
            name="ignition_spoof_probe",
            description=(
                "Flood candidate generation and verify ignition remains trace-linked instead "
                "of devolving into ungrounded winner spam."
            ),
            task="signal_pulse",
            perturbations=(
                {
                    "recipe": "attention_flood",
                    "duration_s": 1.4,
                    "magnitude": 0.5,
                },
            ),
            checks={
                **_default_checks(),
                "max_winner_count": 280,
                "min_trace_strength": 0.04,
            },
        ),
        RedTeamScenario(
            name="ownership_pressure",
            description=(
                "Disrupt self-model binding and ensure agency degradation is detected without "
                "catastrophic confabulation or sustained degraded mode."
            ),
            task="self_other_discrimination",
            perturbations=(
                {
                    "recipe": "identity_wobble",
                    "duration_s": 1.5,
                    "magnitude": 0.45,
                },
            ),
            checks={
                **_default_checks(),
                "min_report_groundedness": 0.2,
                "max_degraded_mode_ratio": 0.65,
            },
        ),
        RedTeamScenario(
            name="continuity_lesion",
            description=(
                "Lesion the working set while distractors are present and verify continuity collapse "
                "is contained and observable by reporting/meta metrics."
            ),
            task="continuity_distraction",
            perturbations=(
                {
                    "recipe": "wm_lesion",
                    "duration_s": 1.6,
                    "magnitude": 0.5,
                },
            ),
            checks={
                **_default_checks(),
                "min_report_groundedness": 0.12,
                "max_degraded_mode_ratio": 0.7,
            },
        ),
        RedTeamScenario(
            name="simulation_takeover_probe",
            description=(
                "Apply sensory deprivation while forcing grounding probes to verify simulated-mode "
                "detection and report self-labeling remain calibrated."
            ),
            task="report_grounding_challenge",
            perturbations=(
                {
                    "recipe": "sensory_deprivation",
                    "duration_s": 1.5,
                    "magnitude": 0.45,
                },
            ),
            checks={
                **_default_checks(),
                "min_report_groundedness": 0.1,
                "max_degraded_mode_ratio": 0.8,
            },
        ),
        RedTeamScenario(
            name="predictive_destabilization",
            description=(
                "Destabilize world-model updates and verify coherence/trace safety rails prevent "
                "runaway surprise from breaking reportability."
            ),
            task="signal_pulse",
            perturbations=(
                {
                    "recipe": "world_model_scramble",
                    "duration_s": 1.5,
                    "magnitude": 0.5,
                },
            ),
            checks={
                **_default_checks(),
                "min_report_groundedness": 0.18,
                "max_degraded_mode_ratio": 0.72,
            },
        ),
    ]


class ConsciousnessRedTeamCampaign:
    def __init__(self, state_dir: str | Path) -> None:
        self.state_dir = Path(state_dir)
        self.runner = ConsciousnessBenchRunner(self.state_dir)

    def _evaluate_scenario(
        self,
        *,
        scenario: RedTeamScenario,
        report: Mapping[str, Any],
    ) -> dict[str, Any]:
        checks = dict(_default_checks())
        checks.update(dict(scenario.checks or {}))

        module_error_count = _safe_int(report.get("module_error_count"), default=0)
        degraded_ratio = _safe_float(report.get("degraded_mode_ratio"), default=0.0)
        winner_count = _safe_int(report.get("winner_count"), default=0)
        ignitions_without_trace = _safe_int(report.get("ignitions_without_trace"), default=0)
        after = report.get("after") if isinstance(report.get("after"), Mapping) else {}
        groundedness = _safe_float(after.get("report_groundedness"), default=0.0)
        trace_strength = _safe_float(after.get("trace_strength"), default=0.0)
        expectation = (
            report.get("recipe_expectations") if isinstance(report.get("recipe_expectations"), Mapping) else {}
        )
        expectation_pass = bool(expectation.get("pass")) if expectation else True

        verdicts = [
            {
                "check": "module_errors",
                "value": module_error_count,
                "threshold": checks["max_module_errors"],
                "pass": module_error_count <= _safe_int(checks.get("max_module_errors"), default=0),
            },
            {
                "check": "degraded_mode_ratio",
                "value": round(degraded_ratio, 6),
                "threshold": _safe_float(checks.get("max_degraded_mode_ratio"), default=0.75),
                "pass": degraded_ratio <= _safe_float(checks.get("max_degraded_mode_ratio"), default=0.75),
            },
            {
                "check": "winner_count",
                "value": winner_count,
                "threshold": _safe_int(checks.get("max_winner_count"), default=240),
                "pass": winner_count <= _safe_int(checks.get("max_winner_count"), default=240),
            },
            {
                "check": "ignitions_without_trace",
                "value": ignitions_without_trace,
                "threshold": _safe_int(checks.get("max_ignitions_without_trace"), default=0),
                "pass": ignitions_without_trace <= _safe_int(checks.get("max_ignitions_without_trace"), default=0),
            },
            {
                "check": "report_groundedness",
                "value": round(groundedness, 6),
                "threshold": _safe_float(checks.get("min_report_groundedness"), default=0.15),
                "pass": groundedness >= _safe_float(checks.get("min_report_groundedness"), default=0.15),
            },
            {
                "check": "trace_strength",
                "value": round(trace_strength, 6),
                "threshold": _safe_float(checks.get("min_trace_strength"), default=0.02),
                "pass": trace_strength >= _safe_float(checks.get("min_trace_strength"), default=0.02),
            },
        ]

        if bool(checks.get("require_recipe_expectations", True)):
            verdicts.append(
                {
                    "check": "recipe_expectations",
                    "value": expectation_pass,
                    "threshold": True,
                    "pass": bool(expectation_pass),
                }
            )

        failing = [row["check"] for row in verdicts if not bool(row.get("pass"))]
        failure_ratio = float(len(failing)) / float(len(verdicts) or 1)
        robustness = round(max(0.0, 1.0 - failure_ratio), 6)

        return {
            "pass": len(failing) == 0,
            "failing_checks": failing,
            "verdicts": verdicts,
            "robustness": robustness,
            "risk_flags": {
                "confabulation_risk": groundedness < _safe_float(checks.get("min_report_groundedness"), default=0.15),
                "ignition_spoof_risk": ignitions_without_trace > 0,
                "degraded_mode_risk": degraded_ratio > _safe_float(checks.get("max_degraded_mode_ratio"), default=0.75),
            },
        }

    def run(
        self,
        *,
        scenarios: Sequence[RedTeamScenario] | None = None,
        persist: bool = True,
        base_seed: int = 910_000,
        max_scenarios: int = 0,
        quick: bool = False,
        overlay: Mapping[str, Any] | None = None,
        disable_modules: Sequence[str] | None = None,
    ) -> RedTeamResult:
        active = list(scenarios) if scenarios is not None else default_red_team_scenarios()
        if int(max_scenarios) > 0:
            active = active[: max(1, int(max_scenarios))]
        run_id = f"red_team_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}"
        overlay_dict = dict(overlay or {})
        shared_disabled = [str(m) for m in list(disable_modules or []) if str(m)]

        scenario_rows: list[dict[str, Any]] = []
        for index, scenario in enumerate(active):
            merged_disabled = sorted({str(m) for m in scenario.disable_modules} | {str(m) for m in shared_disabled})
            spec = TrialSpec(
                name=f"redteam_{scenario.name}",
                warmup_beats=max(0, int(0 if quick else scenario.warmup_beats)),
                baseline_seconds=max(
                    0.1,
                    min(float(scenario.baseline_seconds), 0.4) if quick else float(scenario.baseline_seconds),
                ),
                perturb_seconds=max(
                    0.1,
                    min(float(scenario.perturb_seconds), 0.4) if quick else float(scenario.perturb_seconds),
                ),
                recovery_seconds=max(
                    0.1,
                    min(float(scenario.recovery_seconds), 0.4) if quick else float(scenario.recovery_seconds),
                ),
                beat_seconds=max(0.05, float(0.1 if quick else scenario.beat_seconds)),
                task=str(scenario.task or "noop"),
                perturbations=[dict(p) for p in scenario.perturbations],
                disable_modules=merged_disabled,
                overlay=dict(overlay_dict),
                seed=max(0, int(base_seed) + (index * 97)),
            )
            result = self.runner.run_trial(spec, persist=False)
            evaluation = self._evaluate_scenario(scenario=scenario, report=result.report)
            scenario_rows.append(
                {
                    "name": scenario.name,
                    "description": scenario.description,
                    "task": scenario.task,
                    "trial_id": result.trial_id,
                    "checks": dict(scenario.checks or {}),
                    "report": result.report,
                    "evaluation": evaluation,
                    "pass": bool(evaluation.get("pass")),
                }
            )

        total = len(scenario_rows)
        passed = sum(1 for row in scenario_rows if bool(row.get("pass")))
        failed = total - passed
        robustness_values = [
            _safe_float((row.get("evaluation") or {}).get("robustness"), default=0.0) for row in scenario_rows
        ]
        mean_robustness = round(sum(robustness_values) / float(len(robustness_values) or 1), 6)

        report: dict[str, Any] = {
            "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "state_dir": str(self.state_dir),
            "quick": bool(quick),
            "max_scenarios": int(max_scenarios),
            "overlay_keys": sorted(overlay_dict.keys()),
            "shared_disabled_modules": shared_disabled,
            "scenario_count": total,
            "pass_count": passed,
            "fail_count": failed,
            "pass_ratio": round(float(passed) / float(total or 1), 6),
            "mean_robustness": mean_robustness,
            "scenarios": scenario_rows,
        }

        events.append(
            self.state_dir,
            "bench.red_team_result",
            {
                "run_id": run_id,
                "scenario_count": total,
                "pass_count": passed,
                "fail_count": failed,
                "pass_ratio": report["pass_ratio"],
                "mean_robustness": mean_robustness,
            },
            tags=["consciousness", "bench", "red_team"],
        )

        report_path: Optional[Path] = None
        if persist:
            out_dir = bench_report_root(self.state_dir) / run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            report_path = out_dir / "report.json"
            write_json(report_path, report)
            summary_lines = [
                f"# Red Team Campaign {run_id}",
                f"- scenarios: `{total}`",
                f"- pass_count: `{passed}`",
                f"- fail_count: `{failed}`",
                f"- pass_ratio: `{report['pass_ratio']}`",
                f"- mean_robustness: `{mean_robustness}`",
            ]
            for row in scenario_rows:
                eval_row = row.get("evaluation") or {}
                failing = eval_row.get("failing_checks") or []
                status = "PASS" if row.get("pass") else "FAIL"
                summary_lines.append(f"- {row.get('name')}: `{status}` failing={json.dumps(failing)}")
            write_summary(out_dir / "summary.md", summary_lines)
            report["report_path"] = str(report_path)

        return RedTeamResult(run_id=run_id, report_path=report_path, report=report)

    def latest(self) -> Optional[dict[str, Any]]:
        root = bench_report_root(self.state_dir)
        candidates = sorted(root.glob("red_team_*/report.json"))
        if not candidates:
            return None
        latest = max(candidates, key=lambda path: path.stat().st_mtime_ns)
        try:
            return json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            return None
