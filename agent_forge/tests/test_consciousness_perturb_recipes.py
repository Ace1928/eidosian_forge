from __future__ import annotations

from pathlib import Path

from agent_forge.consciousness.bench.trials import ConsciousnessBenchRunner, TrialSpec
from agent_forge.consciousness.perturb import (
    PerturbationRecipe,
    apply_recipe,
    available_recipes,
    evaluate_expected_signatures,
    recipe_from_name,
)


def test_recipe_catalog_contains_expected_recipes() -> None:
    names = set(available_recipes())
    assert "sensory_deprivation" in names
    assert "attention_flood" in names
    assert "identity_wobble" in names
    assert "wm_lesion" in names
    assert "dopamine_spike" in names
    assert "gw_bottleneck_strain" in names
    assert "world_model_scramble" in names


def test_recipe_expected_signature_evaluation() -> None:
    recipe = recipe_from_name("wm_lesion", duration_s=1.0, magnitude=0.4)
    assert isinstance(recipe, PerturbationRecipe)
    assert "continuity_delta" in recipe.expected_signatures

    observed = {
        "continuity_delta": -0.22,
        "perspective_coherence_delta": -0.08,
        "coherence_delta": -0.03,
    }
    eval_out = evaluate_expected_signatures(recipe.expected_signatures, observed, tolerance=0.02)
    assert eval_out["defined"] is True
    assert eval_out["pass"] is True
    assert len(eval_out["checks"]) == len(recipe.expected_signatures)


def test_apply_recipe_registers_all_perturbations() -> None:
    class _KernelStub:
        def __init__(self) -> None:
            self.payloads: list[dict] = []

        def register_perturbation(self, payload: dict) -> None:
            self.payloads.append(dict(payload))

    recipe = recipe_from_name("attention_flood", duration_s=1.2, magnitude=0.45)
    assert isinstance(recipe, PerturbationRecipe)

    stub = _KernelStub()
    result = apply_recipe(stub, recipe)

    assert result.applied is True
    assert result.details.get("recipe") == "attention_flood"
    assert result.details.get("count") == len(recipe.perturbations)
    assert len(stub.payloads) == len(recipe.perturbations)


def test_bench_runner_expands_recipe_entries(tmp_path: Path) -> None:
    base = tmp_path / "state"
    runner = ConsciousnessBenchRunner(base)
    spec = TrialSpec(
        name="recipe-smoke",
        warmup_beats=1,
        baseline_seconds=0.2,
        perturb_seconds=0.2,
        recovery_seconds=0.2,
        beat_seconds=0.1,
        perturbations=[
            {
                "recipe": "wm_lesion",
                "duration_s": 0.2,
                "magnitude": 0.5,
            }
        ],
    )

    result = runner.run_trial(spec, persist=False)
    report = result.report

    recipes = report.get("recipes") or []
    assert recipes
    assert recipes[0].get("name") == "wm_lesion"
    expectations = report.get("recipe_expectations") or {}
    assert expectations.get("defined") is True
    assert isinstance(expectations.get("checks"), list)
