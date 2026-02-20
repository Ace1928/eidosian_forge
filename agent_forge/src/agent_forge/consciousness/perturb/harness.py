from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

from .library import Perturbation, PerturbationRecipe, to_payload


@dataclass
class PerturbationResult:
    applied: bool
    details: Dict[str, Any]


def apply_perturbation(_: Any, perturbation: Perturbation) -> PerturbationResult:
    payload = to_payload(perturbation)
    register = getattr(_, "register_perturbation", None)
    if callable(register):
        try:
            register(payload)
        except Exception:
            pass

    return PerturbationResult(
        applied=True,
        details=payload,
    )


def apply_perturbations(_: Any, perturbations: Sequence[Perturbation]) -> PerturbationResult:
    payloads: list[dict[str, Any]] = []
    applied = False
    for perturbation in perturbations:
        result = apply_perturbation(_, perturbation)
        payloads.append(dict(result.details))
        if result.applied:
            applied = True
    return PerturbationResult(
        applied=bool(applied),
        details={
            "count": len(payloads),
            "perturbations": payloads,
        },
    )


def apply_recipe(_: Any, recipe: PerturbationRecipe) -> PerturbationResult:
    result = apply_perturbations(_, list(recipe.perturbations))
    details = dict(result.details)
    details.update(
        {
            "recipe": recipe.name,
            "description": recipe.description,
            "expected_signatures": dict(recipe.expected_signatures),
        }
    )
    return PerturbationResult(applied=result.applied, details=details)
