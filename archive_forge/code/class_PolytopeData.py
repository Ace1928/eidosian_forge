from __future__ import annotations
from copy import copy
from dataclasses import dataclass, field
from itertools import combinations
import numpy as np
from qiskit.exceptions import QiskitError
from .utilities import EPSILON
@dataclass
class PolytopeData:
    """
    The raw data of a union of convex polytopes.
    """
    convex_subpolytopes: list[ConvexPolytopeData]