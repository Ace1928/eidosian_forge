from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from agent_forge.core import events, workspace

from ..kernel import ConsciousnessKernel
from ..metrics import (
    coherence_from_workspace_summary,
    directionality_asymmetry,
    effective_connectivity,
    response_complexity,
    self_stability,
)
from ..perturb import evaluate_expected_signatures, perturbations_from_recipe, to_payload
from ..types import clamp01
from .reporting import spec_hash, trial_output_dir, write_json, write_jsonl, write_summary
from .scoring import compute_trial_deltas, composite_trial_score
from .tasks import apply_task_stage, resolve_task


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_int(value: Any, default: int = 0, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def _safe_float(value: Any, default: float = 0.0, minimum: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(float(minimum), parsed)


def _latest_metric(items: list[dict[str, Any]], key: str) -> Optional[float]:
    for evt in reversed(items):
        if str(evt.get("type") or "") != "metrics.sample":
            continue
        data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
        if str(data.get("key") or "") != key:
            continue
        try:
            return float(data.get("value"))
        except (TypeError, ValueError):
            return None
    return None


def _latest_event_data(items: list[dict[str, Any]], etype: str) -> dict[str, Any]:
    for evt in reversed(items):
        if str(evt.get("type") or "") != etype:
            continue
        data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
        return dict(data)
    return {}


@dataclass
class TrialSpec:
    name: str = "default"
    warmup_beats: int = 2
    baseline_seconds: float = 2.0
    perturb_seconds: float = 2.0
    recovery_seconds: float = 2.0
    beat_seconds: float = 0.25
    task: str | None = "noop"
    perturbations: list[dict[str, Any]] = field(default_factory=list)
    disable_modules: list[str] = field(default_factory=list)
    seed: int = 1337

    def normalized(self) -> dict[str, Any]:
        return {
            "name": str(self.name or "default"),
            "warmup_beats": _safe_int(self.warmup_beats, default=2, minimum=0),
            "baseline_seconds": _safe_float(self.baseline_seconds, default=2.0, minimum=0.0),
            "perturb_seconds": _safe_float(self.perturb_seconds, default=2.0, minimum=0.0),
            "recovery_seconds": _safe_float(self.recovery_seconds, default=2.0, minimum=0.0),
            "beat_seconds": max(0.05, _safe_float(self.beat_seconds, default=0.25, minimum=0.05)),
            "task": str(self.task or "noop"),
            "perturbations": [dict(p) for p in self.perturbations if isinstance(p, Mapping)],
            "disable_modules": sorted({str(m) for m in self.disable_modules if str(m)}),
            "seed": _safe_int(self.seed, default=1337, minimum=0),
        }


@dataclass
class TrialRunResult:
    trial_id: str
    output_dir: Optional[Path]
    report: Dict[str, Any]


class ConsciousnessBenchRunner:
    def __init__(self, state_dir: str | Path) -> None:
        self.state_dir = Path(state_dir)

    def _beats_for(self, seconds: float, beat_seconds: float) -> int:
        duration = max(0.0, float(seconds))
        beat = max(0.05, float(beat_seconds))
        if duration <= 0.0:
            return 0
        return max(1, int(round(duration / beat)))

    def _snapshot(self, recent_events: list[dict[str, Any]]) -> dict[str, Any]:
        ws = workspace.summary(self.state_dir, limit=500, window_seconds=1.0, min_sources=3)
        coh = coherence_from_workspace_summary(ws)
        rci = response_complexity(recent_events[-300:])
        conn = effective_connectivity(recent_events[-500:])
        dirn = directionality_asymmetry(recent_events[-500:])
        stab = self_stability(recent_events[-500:])
        phenom = _latest_event_data(recent_events, "phenom.snapshot")
        return {
            "workspace": ws,
            "coherence_ratio": float(coh.get("coherence_ratio") or 0.0),
            "ignition_density": float(coh.get("ignition_density") or 0.0),
            "rci": rci,
            "connectivity": conn,
            "directionality": dirn,
            "self_stability": stab,
            "phenomenology": phenom,
            "agency": _latest_metric(recent_events, "consciousness.agency"),
            "boundary_stability": _latest_metric(recent_events, "consciousness.boundary_stability"),
            "world_prediction_error": _latest_metric(recent_events, "consciousness.world.prediction_error"),
            "report_groundedness": _latest_metric(recent_events, "consciousness.report.groundedness"),
            "trace_strength": _latest_metric(recent_events, "consciousness.ignition.trace_strength"),
        }

    def _run_stage(
        self,
        *,
        kernel: ConsciousnessKernel,
        task_name: str,
        stage: str,
        beats: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        task = resolve_task(task_name)
        for i in range(max(0, int(beats))):
            apply_task_stage(
                state_dir=self.state_dir,
                task=task,
                stage=stage,
                beat=i,
            )
            kernel.tick()
            recent = events.iter_events(self.state_dir, limit=600)
            rows.append(
                {
                    "ts": _now_iso(),
                    "stage": stage,
                    "beat": int(i),
                    **self._snapshot(recent),
                }
            )
        return rows

    def run_trial(
        self,
        spec: TrialSpec,
        *,
        kernel: Optional[ConsciousnessKernel] = None,
        persist: bool = True,
    ) -> TrialRunResult:
        norm = spec.normalized()
        kernel = kernel or ConsciousnessKernel(self.state_dir, seed=int(norm["seed"]))
        before_events = events.iter_events(self.state_dir, limit=None)
        before_count = len(before_events)
        before = self._snapshot(before_events[-800:])

        original_disable = list(kernel.config.get("disable_modules") or [])
        disabled = sorted({str(x) for x in original_disable} | set(norm["disable_modules"]))
        kernel.config["disable_modules"] = disabled

        beat_seconds = float(norm["beat_seconds"])
        warmup_beats = int(norm["warmup_beats"])
        baseline_beats = self._beats_for(float(norm["baseline_seconds"]), beat_seconds)
        perturb_beats = self._beats_for(float(norm["perturb_seconds"]), beat_seconds)
        recovery_beats = self._beats_for(float(norm["recovery_seconds"]), beat_seconds)

        stage_rows: list[dict[str, Any]] = []
        perturbation_rows: list[dict[str, Any]] = []
        recipe_rows: list[dict[str, Any]] = []
        recipe_expected_signatures: dict[str, str] = {}
        trial_corr = uuid.uuid4().hex
        try:
            for _ in range(warmup_beats):
                kernel.tick()

            stage_rows.extend(
                self._run_stage(
                    kernel=kernel,
                    task_name=str(norm["task"]),
                    stage="baseline",
                    beats=baseline_beats,
                )
            )

            expanded_perturbations: list[dict[str, Any]] = []
            for raw in norm["perturbations"]:
                recipe, recipe_perturbs = perturbations_from_recipe(raw)
                if recipe is not None:
                    recipe_rows.append(
                        {
                            "name": recipe.name,
                            "description": recipe.description,
                            "expected_signatures": dict(recipe.expected_signatures),
                        }
                    )
                    recipe_expected_signatures.update(dict(recipe.expected_signatures))
                    expanded_perturbations.extend([to_payload(p) for p in recipe_perturbs])
                else:
                    expanded_perturbations.append(dict(raw))

            for raw in expanded_perturbations:
                payload = {
                    "id": str(raw.get("id") or uuid.uuid4().hex),
                    "kind": str(raw.get("kind") or "noise"),
                    "target": str(raw.get("target") or "attention"),
                    "magnitude": clamp01(raw.get("magnitude"), default=0.2),
                    "duration_s": _safe_float(raw.get("duration_s"), default=float(norm["perturb_seconds"]), minimum=0.0),
                    "meta": dict(raw.get("meta") or {}),
                    "ts": _now_iso(),
                }
                perturbation_rows.append(payload)
                events.append(
                    self.state_dir,
                    "perturb.inject",
                    dict(payload),
                    tags=["consciousness", "perturb", "bench"],
                    corr_id=trial_corr,
                    parent_id=trial_corr,
                )
                kernel.register_perturbation(payload)

            stage_rows.extend(
                self._run_stage(
                    kernel=kernel,
                    task_name=str(norm["task"]),
                    stage="perturb",
                    beats=perturb_beats,
                )
            )
            stage_rows.extend(
                self._run_stage(
                    kernel=kernel,
                    task_name=str(norm["task"]),
                    stage="recovery",
                    beats=recovery_beats,
                )
            )
        finally:
            kernel.config["disable_modules"] = original_disable

        all_events = events.iter_events(self.state_dir, limit=None)
        after = self._snapshot(all_events[-1000:])
        window_events = all_events[before_count:]

        deltas = compute_trial_deltas(before, after)
        score = composite_trial_score(deltas)
        expectation_eval = evaluate_expected_signatures(
            recipe_expected_signatures,
            deltas,
            tolerance=0.01,
        )
        trial_hash = spec_hash(norm)
        trial_spec = {
            **norm,
            "warmup_beats": warmup_beats,
            "baseline_beats": baseline_beats,
            "perturb_beats": perturb_beats,
            "recovery_beats": recovery_beats,
        }
        trial_id = (
            f"trial_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_"
            f"{str(norm['name'])}_{trial_hash[:8]}"
        )

        report: dict[str, Any] = {
            "trial_id": trial_id,
            "timestamp": _now_iso(),
            "state_dir": str(self.state_dir),
            "spec_hash": trial_hash,
            "spec": trial_spec,
            "before": before,
            "after": after,
            "deltas": deltas,
            "composite_score": score,
            "perturbations": perturbation_rows,
            "recipes": recipe_rows,
            "recipe_expectations": expectation_eval,
            "events_window_count": len(window_events),
            "stage_rows": len(stage_rows),
        }

        events.append(
            self.state_dir,
            "bench.trial_result",
            {
                "trial_id": trial_id,
                "spec_hash": trial_hash,
                "name": str(norm["name"]),
                "composite_score": score,
                "deltas": deltas,
                "recipes": recipe_rows,
                "recipe_expectations": expectation_eval,
                "events_window_count": len(window_events),
                "stage_rows": len(stage_rows),
            },
            tags=["consciousness", "bench", "trial"],
            corr_id=trial_corr,
            parent_id=trial_corr,
        )

        output_dir: Optional[Path] = None
        if persist:
            output_dir = trial_output_dir(
                self.state_dir,
                name=str(norm["name"]),
                trial_hash=trial_hash,
            )
            write_json(output_dir / "spec.json", trial_spec)
            write_jsonl(output_dir / "metrics.jsonl", stage_rows)
            write_jsonl(output_dir / "events_window.jsonl", window_events)
            write_json(output_dir / "report.json", report)
            write_summary(
                output_dir / "summary.md",
                [
                    f"# Trial {trial_id}",
                    f"- spec_hash: `{trial_hash}`",
                    f"- composite_score: `{score}`",
                    f"- ignition_delta: `{deltas.get('ignition_delta')}`",
                    f"- coherence_delta: `{deltas.get('coherence_delta')}`",
                    f"- rci_delta: `{deltas.get('rci_delta')}`",
                    f"- trace_strength_delta: `{deltas.get('trace_strength_delta')}`",
                    f"- recipe_expectations: `{expectation_eval.get('pass')}`",
                    f"- artifacts: `spec.json`, `metrics.jsonl`, `events_window.jsonl`, `report.json`",
                ],
            )
            report["output_dir"] = str(output_dir)

        return TrialRunResult(trial_id=trial_id, output_dir=output_dir, report=report)
