from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from .library import Perturbation


@dataclass
class PerturbationResult:
    applied: bool
    details: Dict[str, Any]


def apply_perturbation(_: Any, perturbation: Perturbation) -> PerturbationResult:
    payload = {
        "kind": perturbation.kind,
        "target": perturbation.target,
        "magnitude": float(perturbation.magnitude),
        "duration_s": float(perturbation.duration_s),
        "meta": dict(perturbation.meta),
    }
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
