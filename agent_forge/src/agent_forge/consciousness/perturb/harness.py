from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from .library import Perturbation


@dataclass
class PerturbationResult:
    applied: bool
    details: Dict[str, Any]


def apply_perturbation(_: Any, perturbation: Perturbation) -> PerturbationResult:
    # This harness intentionally starts as a no-op adapter.
    # Modules can opt in by reading the perturbation event stream.
    return PerturbationResult(
        applied=True,
        details={
            "kind": perturbation.kind,
            "target": perturbation.target,
            "magnitude": perturbation.magnitude,
            "duration_s": perturbation.duration_s,
            "meta": dict(perturbation.meta),
        },
    )
