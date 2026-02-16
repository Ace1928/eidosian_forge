from __future__ import annotations

from pathlib import Path

from agent_forge.consciousness.bench.ablations import ConsciousnessAblationMatrix
from agent_forge.consciousness.bench.golden import evaluate_variant_golden
from agent_forge.consciousness.bench.trials import TrialSpec
from agent_forge.core import events


def test_evaluate_variant_golden_structure() -> None:
    out = evaluate_variant_golden(
        "no_competition",
        {
            "trace_strength_delta_vs_full": -0.2,
            "coherence_ratio_delta_vs_full": -0.05,
        },
    )
    assert out["variant"] == "no_competition"
    assert out["defined"] is True
    assert isinstance(out["checks"], list)
    assert all("pass" in row for row in out["checks"])


def test_ablation_matrix_runs_and_emits_event(tmp_path: Path) -> None:
    base = tmp_path / "state"
    matrix = ConsciousnessAblationMatrix(base)
    spec = TrialSpec(
        name="ablation-smoke",
        warmup_beats=1,
        baseline_seconds=0.2,
        perturb_seconds=0.2,
        recovery_seconds=0.2,
        beat_seconds=0.1,
        perturbations=[
            {
                "kind": "noise",
                "target": "attention",
                "magnitude": 0.2,
                "duration_s": 0.2,
            }
        ],
    )
    result = matrix.run(
        base_spec=spec,
        variants={"no_competition": ["workspace_competition"]},
        persist=False,
    )

    assert result.run_id
    matrix_report = result.report.get("matrix") or {}
    assert "full" in matrix_report
    assert "variants" in matrix_report
    assert "no_competition" in (matrix_report.get("variants") or {})

    all_events = events.iter_events(base, limit=None)
    assert any(evt.get("type") == "bench.ablation_result" for evt in all_events)
