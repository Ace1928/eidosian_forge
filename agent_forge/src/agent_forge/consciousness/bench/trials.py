from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from agent_forge.core import events, workspace

from ..index import EventIndex, build_index
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


def _latest_metric(index: EventIndex, key: str) -> Optional[float]:
    for evt in reversed(index.by_type.get("metrics.sample") or []):
        data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
        if str(data.get("key") or "") != key:
            continue
        try:
            return float(data.get("value"))
        except (TypeError, ValueError):
            return None
    return None


def _latest_event_data(index: EventIndex, etype: str) -> dict[str, Any]:
    evt = index.latest_by_type.get(str(etype))
    if evt is not None:
        data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
        return dict(data)
    return {}


def _slice_events_by_event_ids(
    all_events: list[dict[str, Any]],
    *,
    start_event_id: str,
    end_event_id: str,
) -> list[dict[str, Any]]:
    if not start_event_id or not end_event_id:
        return []
    window: list[dict[str, Any]] = []
    capturing = False
    for evt in all_events:
        event_id = str(evt.get("event_id") or "")
        if not capturing and event_id == start_event_id:
            capturing = True
        if capturing:
            window.append(evt)
        if capturing and event_id == end_event_id:
            break
    if not window:
        return []
    if str(window[-1].get("event_id") or "") != end_event_id:
        return []
    return window


@dataclass
class TrialSpec:
    name: str = "default"
    warmup_beats: int = 2
    baseline_seconds: float = 2.0
    perturb_seconds: float = 2.0
    recovery_seconds: float = 2.0
    baseline_beats: int = 0
    perturb_beats: int = 0
    recovery_beats: int = 0
    beat_seconds: float = 0.25
    task: str | None = "noop"
    perturbations: list[dict[str, Any]] = field(default_factory=list)
    disable_modules: list[str] = field(default_factory=list)
    overlay: dict[str, Any] = field(default_factory=dict)
    seed: int = 1337

    def normalized(self) -> dict[str, Any]:
        return {
            "name": str(self.name or "default"),
            "warmup_beats": _safe_int(self.warmup_beats, default=2, minimum=0),
            "baseline_seconds": _safe_float(self.baseline_seconds, default=2.0, minimum=0.0),
            "perturb_seconds": _safe_float(self.perturb_seconds, default=2.0, minimum=0.0),
            "recovery_seconds": _safe_float(self.recovery_seconds, default=2.0, minimum=0.0),
            "baseline_beats": _safe_int(self.baseline_beats, default=0, minimum=0),
            "perturb_beats": _safe_int(self.perturb_beats, default=0, minimum=0),
            "recovery_beats": _safe_int(self.recovery_beats, default=0, minimum=0),
            "beat_seconds": max(0.05, _safe_float(self.beat_seconds, default=0.25, minimum=0.05)),
            "task": str(self.task or "noop"),
            "perturbations": [dict(p) for p in self.perturbations if isinstance(p, Mapping)],
            "disable_modules": sorted({str(m) for m in self.disable_modules if str(m)}),
            "overlay": dict(self.overlay or {}),
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
        index = build_index(recent_events)
        phenom = _latest_event_data(index, "phenom.snapshot")
        return {
            "workspace": ws,
            "coherence_ratio": float(coh.get("coherence_ratio") or 0.0),
            "ignition_density": float(coh.get("ignition_density") or 0.0),
            "rci": rci,
            "connectivity": conn,
            "directionality": dirn,
            "self_stability": stab,
            "phenomenology": phenom,
            "agency": _latest_metric(index, "consciousness.agency"),
            "boundary_stability": _latest_metric(index, "consciousness.boundary_stability"),
            "world_prediction_error": _latest_metric(index, "consciousness.world.prediction_error"),
            "report_groundedness": _latest_metric(index, "consciousness.report.groundedness"),
            "trace_strength": _latest_metric(index, "consciousness.ignition.trace_strength"),
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
        trial_hash = spec_hash(norm)
        trial_id = (
            f"trial_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_"
            f"{str(norm['name'])}_{trial_hash[:8]}"
        )
        trial_corr = uuid.uuid4().hex
        kernel = kernel or ConsciousnessKernel(
            self.state_dir,
            config=dict(norm["overlay"] or {}),
            seed=int(norm["seed"]),
            respect_tuned_overlay=False,
        )
        before_events = events.iter_events(self.state_dir, limit=None)
        before_count = len(before_events)
        before = self._snapshot(before_events[-800:])

        original_disable = list(kernel.config.get("disable_modules") or [])
        original_runtime_overrides = copy.deepcopy(
            getattr(kernel, "_runtime_overrides", {})
        )
        disabled = sorted({str(x) for x in original_disable} | set(norm["disable_modules"]))
        kernel.config["disable_modules"] = disabled
        if norm["overlay"]:
            kernel.set_runtime_overrides(dict(norm["overlay"]))

        beat_seconds = float(norm["beat_seconds"])
        warmup_beats = int(norm["warmup_beats"])
        baseline_beats = (
            int(norm["baseline_beats"])
            if int(norm["baseline_beats"]) > 0
            else self._beats_for(float(norm["baseline_seconds"]), beat_seconds)
        )
        perturb_beats = (
            int(norm["perturb_beats"])
            if int(norm["perturb_beats"]) > 0
            else self._beats_for(float(norm["perturb_seconds"]), beat_seconds)
        )
        recovery_beats = (
            int(norm["recovery_beats"])
            if int(norm["recovery_beats"]) > 0
            else self._beats_for(float(norm["recovery_seconds"]), beat_seconds)
        )

        stage_rows: list[dict[str, Any]] = []
        perturbation_rows: list[dict[str, Any]] = []
        recipe_rows: list[dict[str, Any]] = []
        recipe_expected_signatures: dict[str, str] = {}
        trial_start_event = events.append(
            self.state_dir,
            "bench.trial_start",
            {
                "trial_id": trial_id,
                "spec_hash": trial_hash,
                "name": str(norm["name"]),
                "warmup_beats": warmup_beats,
                "baseline_beats": baseline_beats,
                "perturb_beats": perturb_beats,
                "recovery_beats": recovery_beats,
            },
            tags=["consciousness", "bench", "trial"],
            corr_id=trial_corr,
            parent_id=trial_corr,
        )
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
            if norm["overlay"]:
                kernel.set_runtime_overrides(original_runtime_overrides)

        all_events = events.iter_events(self.state_dir, limit=None)
        after = self._snapshot(all_events[-1000:])
        analysis_window_events = all_events[before_count:]
        threshold = float(kernel.config.get("competition_trace_strength_threshold", 0.45))
        index = build_index(analysis_window_events)

        event_type_counts = {
            etype: int(len(rows))
            for etype, rows in index.by_type.items()
            if str(etype)
        }
        module_error_count = int(len(index.by_type.get("consciousness.module_error") or []))
        meta_events = index.by_type.get("meta.state_estimate") or []
        meta_total = int(len(meta_events))
        degraded_count = 0
        for evt in meta_events:
            data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
            if str(data.get("mode") or "").lower() == "degraded":
                degraded_count += 1

        winner_count = int(len(index.broadcasts_by_kind.get("GW_WINNER") or []))
        ignitions_without_trace = 0
        for evt in index.by_type.get("gw.ignite") or []:
            data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
            try:
                trace_strength = float(data.get("trace_strength") or 0.0)
            except (TypeError, ValueError):
                trace_strength = 0.0
            if trace_strength < threshold:
                ignitions_without_trace += 1
        degraded_mode_ratio = (
            round(float(degraded_count) / float(meta_total), 6) if meta_total > 0 else 0.0
        )

        deltas = compute_trial_deltas(before, after)
        score = composite_trial_score(deltas)
        expectation_eval = evaluate_expected_signatures(
            recipe_expected_signatures,
            deltas,
            tolerance=0.01,
        )
        trial_spec = {
            **norm,
            "warmup_beats": warmup_beats,
            "baseline_beats": baseline_beats,
            "perturb_beats": perturb_beats,
            "recovery_beats": recovery_beats,
        }

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
            "events_window_count": len(analysis_window_events),
            "event_type_counts": event_type_counts,
            "module_error_count": module_error_count,
            "degraded_mode_ratio": degraded_mode_ratio,
            "winner_count": winner_count,
            "ignitions_without_trace": ignitions_without_trace,
            "stage_rows": len(stage_rows),
            "capture_method": "before_count_fallback",
            "capture_start_event_id": str(trial_start_event.get("event_id") or ""),
            "capture_end_event_id": "",
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
                "events_window_count": len(analysis_window_events),
                "module_error_count": module_error_count,
                "degraded_mode_ratio": degraded_mode_ratio,
                "winner_count": winner_count,
                "ignitions_without_trace": ignitions_without_trace,
                "stage_rows": len(stage_rows),
            },
            tags=["consciousness", "bench", "trial"],
            corr_id=trial_corr,
            parent_id=trial_corr,
        )

        trial_end_event = events.append(
            self.state_dir,
            "bench.trial_end",
            {
                "trial_id": trial_id,
                "spec_hash": trial_hash,
                "name": str(norm["name"]),
                "composite_score": score,
            },
            tags=["consciousness", "bench", "trial"],
            corr_id=trial_corr,
            parent_id=trial_corr,
        )

        final_events = events.iter_events(self.state_dir, limit=None)
        marker_window = _slice_events_by_event_ids(
            final_events,
            start_event_id=str(trial_start_event.get("event_id") or ""),
            end_event_id=str(trial_end_event.get("event_id") or ""),
        )
        if marker_window:
            window_events = marker_window
            report["capture_method"] = "event_id_markers"
        else:
            window_events = final_events[before_count:]
            report["capture_method"] = "before_count_fallback"

        report["capture_end_event_id"] = str(trial_end_event.get("event_id") or "")
        report["events_window_count"] = len(window_events)
        report["event_type_counts"] = {
            etype: int(len(rows))
            for etype, rows in build_index(window_events).by_type.items()
            if str(etype)
        }

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
