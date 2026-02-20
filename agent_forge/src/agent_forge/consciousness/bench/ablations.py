from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from agent_forge.core import events

from .golden import evaluate_variant_golden
from .reporting import bench_report_root, write_json, write_summary
from .trials import ConsciousnessBenchRunner, TrialSpec


@dataclass
class AblationResult:
    run_id: str
    report_path: Optional[Path]
    report: dict[str, Any]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _metric(report: Mapping[str, Any], *path: str, default: float = 0.0) -> float:
    cur: Any = report
    for key in path:
        if not isinstance(cur, Mapping):
            return float(default)
        cur = cur.get(key)
    return _safe_float(cur, default=default)


def _comparison(full: Mapping[str, Any], variant: Mapping[str, Any]) -> dict[str, float]:
    return {
        "composite_delta_vs_full": round(
            _metric(variant, "composite_score") - _metric(full, "composite_score"),
            6,
        ),
        "trace_strength_delta_vs_full": round(
            _metric(variant, "after", "trace_strength") - _metric(full, "after", "trace_strength"),
            6,
        ),
        "coherence_ratio_delta_vs_full": round(
            _metric(variant, "after", "coherence_ratio") - _metric(full, "after", "coherence_ratio"),
            6,
        ),
        "rci_v2_delta_vs_full": round(
            _metric(variant, "after", "rci", "rci_v2") - _metric(full, "after", "rci", "rci_v2"),
            6,
        ),
        "connectivity_density_delta_vs_full": round(
            _metric(variant, "after", "connectivity", "density") - _metric(full, "after", "connectivity", "density"),
            6,
        ),
        "self_stability_delta_vs_full": round(
            _metric(variant, "after", "self_stability", "stability_score")
            - _metric(full, "after", "self_stability", "stability_score"),
            6,
        ),
    }


def default_variants() -> dict[str, list[str]]:
    return {
        "no_competition": ["workspace_competition"],
        "no_working_set": ["working_set"],
        "no_self_model_ext": ["self_model_ext"],
        "no_meta": ["meta"],
        "no_world_model": ["world_model"],
    }


class ConsciousnessAblationMatrix:
    def __init__(self, state_dir: str | Path) -> None:
        self.state_dir = Path(state_dir)
        self.runner = ConsciousnessBenchRunner(self.state_dir)

    def run(
        self,
        *,
        base_spec: TrialSpec,
        variants: Mapping[str, list[str]] | None = None,
        persist: bool = True,
    ) -> AblationResult:
        run_id = f"ablation_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}"
        variant_map = dict(variants or default_variants())

        base_norm = base_spec.normalized()
        full_spec = TrialSpec(
            **{
                "name": f"{base_norm['name']}_full",
                "warmup_beats": int(base_norm["warmup_beats"]),
                "baseline_seconds": float(base_norm["baseline_seconds"]),
                "perturb_seconds": float(base_norm["perturb_seconds"]),
                "recovery_seconds": float(base_norm["recovery_seconds"]),
                "beat_seconds": float(base_norm["beat_seconds"]),
                "task": str(base_norm["task"]),
                "perturbations": list(base_norm["perturbations"]),
                "disable_modules": list(base_norm["disable_modules"]),
                "seed": int(base_norm["seed"]),
            }
        )
        full_result = self.runner.run_trial(full_spec, persist=False)
        matrix: dict[str, Any] = {
            "full": full_result.report,
            "variants": {},
            "golden": {},
        }

        for name, extra_disabled in variant_map.items():
            merged_disabled = sorted(
                {str(m) for m in full_spec.disable_modules} | {str(m) for m in list(extra_disabled or [])}
            )
            v_spec = TrialSpec(
                name=f"{base_norm['name']}_{name}",
                warmup_beats=full_spec.warmup_beats,
                baseline_seconds=full_spec.baseline_seconds,
                perturb_seconds=full_spec.perturb_seconds,
                recovery_seconds=full_spec.recovery_seconds,
                beat_seconds=full_spec.beat_seconds,
                task=full_spec.task,
                perturbations=list(full_spec.perturbations),
                disable_modules=merged_disabled,
                seed=full_spec.seed,
            )
            result = self.runner.run_trial(v_spec, persist=False)
            comp = _comparison(full_result.report, result.report)
            golden = evaluate_variant_golden(name, comp)
            matrix["variants"][name] = {
                "disabled_modules": merged_disabled,
                "report": result.report,
                "comparison": comp,
            }
            matrix["golden"][name] = golden

        report = {
            "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "state_dir": str(self.state_dir),
            "base_spec": base_norm,
            "matrix": matrix,
        }

        events.append(
            self.state_dir,
            "bench.ablation_result",
            {
                "run_id": run_id,
                "variant_count": len(matrix["variants"]),
                "golden_pass_count": sum(1 for row in matrix["golden"].values() if row.get("pass")),
                "golden_total": len(matrix["golden"]),
            },
            tags=["consciousness", "bench", "ablation"],
        )

        report_path: Optional[Path] = None
        if persist:
            out_dir = bench_report_root(self.state_dir) / f"{run_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            report_path = out_dir / "ablation_report.json"
            write_json(report_path, report)
            summary_lines = [
                f"# Ablation {run_id}",
                f"- variants: `{len(matrix['variants'])}`",
                f"- golden_pass: `{sum(1 for row in matrix['golden'].values() if row.get('pass'))}/{len(matrix['golden'])}`",
            ]
            for name, row in matrix["golden"].items():
                summary_lines.append(f"- {name}: `{'PASS' if row.get('pass') else 'FAIL'}`")
            write_summary(out_dir / "summary.md", summary_lines)

        return AblationResult(run_id=run_id, report_path=report_path, report=report)

    def latest(self) -> Optional[dict[str, Any]]:
        root = bench_report_root(self.state_dir)
        candidates = sorted(root.glob("ablation_*/ablation_report.json"))
        if not candidates:
            return None
        latest = max(candidates, key=lambda path: path.stat().st_mtime_ns)
        try:
            return json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            return None
