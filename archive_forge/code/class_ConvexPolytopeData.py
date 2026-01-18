from __future__ import annotations
from copy import copy
from dataclasses import dataclass, field
from itertools import combinations
import numpy as np
from qiskit.exceptions import QiskitError
from .utilities import EPSILON
@dataclass
class ConvexPolytopeData:
    """
    The raw data underlying a ConvexPolytope.  Describes a single convex
    polytope, specified by families of `inequalities` and `equalities`, each
    entry of which respectively corresponds to

        inequalities[j][0] + sum_i inequalities[j][i] * xi >= 0

    and

        equalities[j][0] + sum_i equalities[j][i] * xi == 0.
    """
    inequalities: list[list[int]]
    equalities: list[list[int]] = field(default_factory=list)
    name: str = ''