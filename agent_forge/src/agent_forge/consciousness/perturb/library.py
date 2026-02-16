from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class Perturbation:
    kind: str
    target: str
    magnitude: float
    duration_s: float
    meta: Dict[str, Any]


def make_noise(target: str, magnitude: float, duration_s: float = 1.0) -> Perturbation:
    return Perturbation(kind="noise", target=target, magnitude=float(magnitude), duration_s=float(duration_s), meta={})


def make_drop(target: str, duration_s: float = 1.0) -> Perturbation:
    return Perturbation(kind="drop", target=target, magnitude=1.0, duration_s=float(duration_s), meta={})
