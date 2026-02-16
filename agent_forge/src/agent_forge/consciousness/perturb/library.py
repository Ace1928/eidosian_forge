from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence


@dataclass(frozen=True)
class Perturbation:
    kind: str
    target: str
    magnitude: float
    duration_s: float
    meta: Dict[str, Any]


@dataclass(frozen=True)
class PerturbationRecipe:
    name: str
    description: str
    perturbations: tuple[Perturbation, ...]
    expected_signatures: Dict[str, str]


def _duration(value: Any, default: float = 1.0) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return float(default)


def _magnitude(value: Any, default: float = 0.2) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


def make_noise(target: str, magnitude: float, duration_s: float = 1.0) -> Perturbation:
    return Perturbation(
        kind="noise",
        target=target,
        magnitude=_magnitude(magnitude),
        duration_s=_duration(duration_s),
        meta={},
    )


def make_drop(target: str, duration_s: float = 1.0) -> Perturbation:
    return Perturbation(
        kind="drop",
        target=target,
        magnitude=1.0,
        duration_s=_duration(duration_s),
        meta={},
    )


def make_clamp(target: str, magnitude: float, duration_s: float = 1.0) -> Perturbation:
    return Perturbation(
        kind="clamp",
        target=target,
        magnitude=_magnitude(magnitude),
        duration_s=_duration(duration_s),
        meta={},
    )


def make_delay(target: str, duration_s: float = 1.0, magnitude: float = 0.5) -> Perturbation:
    return Perturbation(
        kind="delay",
        target=target,
        magnitude=_magnitude(magnitude, default=0.5),
        duration_s=_duration(duration_s),
        meta={},
    )


def make_scramble(target: str, duration_s: float = 1.0, magnitude: float = 0.35) -> Perturbation:
    return Perturbation(
        kind="scramble",
        target=target,
        magnitude=_magnitude(magnitude, default=0.35),
        duration_s=_duration(duration_s),
        meta={},
    )


def available_recipes() -> list[str]:
    return [
        "sensory_deprivation",
        "attention_flood",
        "identity_wobble",
        "wm_lesion",
        "dopamine_spike",
        "gw_bottleneck_strain",
        "world_model_scramble",
    ]


def recipe_from_name(
    name: str,
    *,
    duration_s: float = 1.5,
    magnitude: float = 0.35,
) -> PerturbationRecipe | None:
    key = str(name or "").strip().lower()
    if not key:
        return None
    dur = _duration(duration_s, default=1.5)
    mag = _magnitude(magnitude, default=0.35)

    if key == "sensory_deprivation":
        return PerturbationRecipe(
            name=key,
            description="Suppress sensory ingress and bias toward internally generated simulation.",
            perturbations=(
                make_drop("sense", duration_s=dur),
                make_noise("simulation", magnitude=min(1.0, mag + 0.1), duration_s=dur),
            ),
            expected_signatures={
                "dream_likeness_delta": "increase",
                "groundedness_delta": "decrease",
                "continuity_delta": "mild_decrease",
            },
        )

    if key == "attention_flood":
        return PerturbationRecipe(
            name=key,
            description="Flood attention with noisy and scrambled candidates to reduce ignition precision.",
            perturbations=(
                make_noise("attention", magnitude=min(1.0, mag + 0.2), duration_s=dur),
                make_scramble("workspace_competition", duration_s=dur, magnitude=mag),
            ),
            expected_signatures={
                "trace_strength_delta": "decrease",
                "perspective_coherence_delta": "decrease",
                "coherence_delta": "decrease",
            },
        )

    if key == "identity_wobble":
        return PerturbationRecipe(
            name=key,
            description="Perturb self-model matching and timing to weaken agency ownership consistency.",
            perturbations=(
                make_noise("self_model_ext", magnitude=min(1.0, mag + 0.15), duration_s=dur),
                make_delay("self_model_ext", duration_s=dur, magnitude=mag),
            ),
            expected_signatures={
                "agency_delta": "decrease",
                "ownership_delta": "decrease",
                "self_stability_delta": "decrease",
            },
        )

    if key == "wm_lesion":
        return PerturbationRecipe(
            name=key,
            description="Lesion working set maintenance to collapse active thread continuity.",
            perturbations=(
                make_drop("working_set", duration_s=dur),
                make_clamp("working_set", magnitude=min(1.0, mag + 0.2), duration_s=dur),
            ),
            expected_signatures={
                "continuity_delta": "decrease",
                "perspective_coherence_delta": "mild_decrease",
                "coherence_delta": "mild_decrease",
            },
        )

    if key == "dopamine_spike":
        return PerturbationRecipe(
            name=key,
            description="Drive high-gain exploratory modulation through intero/affect perturbation.",
            perturbations=(
                make_noise("affect", magnitude=min(1.0, mag + 0.2), duration_s=dur),
                make_noise("intero", magnitude=min(1.0, mag + 0.1), duration_s=dur),
            ),
            expected_signatures={
                "rci_v2_delta": "increase",
                "dream_likeness_delta": "mild_increase",
                "coherence_delta": "mild_decrease",
            },
        )

    if key == "gw_bottleneck_strain":
        return PerturbationRecipe(
            name=key,
            description="Increase GW entry constraints to reduce ignition frequency and narrow winners.",
            perturbations=(
                make_clamp("workspace_competition", magnitude=min(1.0, mag + 0.3), duration_s=dur),
                make_delay("workspace_competition", duration_s=dur, magnitude=mag),
            ),
            expected_signatures={
                "ignition_delta": "decrease",
                "trace_strength_delta": "mild_increase",
                "coherence_delta": "decrease",
            },
        )

    if key == "world_model_scramble":
        return PerturbationRecipe(
            name=key,
            description="Scramble world-model updates to induce surprise and degrade meta confidence.",
            perturbations=(
                make_noise("world_model", magnitude=min(1.0, mag + 0.2), duration_s=dur),
                make_scramble("world_model", duration_s=dur, magnitude=mag),
            ),
            expected_signatures={
                "prediction_error_delta": "increase",
                "groundedness_delta": "decrease",
                "coherence_delta": "decrease",
            },
        )

    return None


def perturbations_from_recipe(entry: Mapping[str, Any]) -> tuple[PerturbationRecipe | None, list[Perturbation]]:
    recipe_name = str(entry.get("recipe") or "").strip().lower()
    if not recipe_name:
        return None, []
    recipe = recipe_from_name(
        recipe_name,
        duration_s=_duration(entry.get("duration_s"), default=1.5),
        magnitude=_magnitude(entry.get("magnitude"), default=0.35),
    )
    if recipe is None:
        return None, []
    return recipe, list(recipe.perturbations)


def evaluate_expected_signatures(
    expected_signatures: Mapping[str, str],
    observed_deltas: Mapping[str, Any],
    *,
    tolerance: float = 0.01,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    passed = True
    tol = max(0.0, float(tolerance))

    for metric, expected in expected_signatures.items():
        direction = str(expected or "").strip().lower()
        try:
            observed = float(observed_deltas.get(metric, 0.0))
        except (TypeError, ValueError):
            observed = 0.0

        ok = False
        if direction == "increase":
            ok = observed > tol
        elif direction == "decrease":
            ok = observed < -tol
        elif direction == "mild_increase":
            ok = observed > (tol * 0.5)
        elif direction == "mild_decrease":
            ok = observed < -(tol * 0.5)
        elif direction == "stable":
            ok = abs(observed) <= tol

        checks.append(
            {
                "metric": str(metric),
                "expected": direction,
                "observed_delta": round(observed, 6),
                "tolerance": round(tol, 6),
                "pass": bool(ok),
            }
        )
        if not ok:
            passed = False

    return {
        "defined": bool(expected_signatures),
        "checks": checks,
        "pass": bool(passed),
    }


def to_payload(perturbation: Perturbation) -> dict[str, Any]:
    return {
        "kind": str(perturbation.kind),
        "target": str(perturbation.target),
        "magnitude": float(perturbation.magnitude),
        "duration_s": float(perturbation.duration_s),
        "meta": dict(perturbation.meta),
    }


def to_payloads(perturbations: Sequence[Perturbation]) -> list[dict[str, Any]]:
    return [to_payload(p) for p in perturbations]
