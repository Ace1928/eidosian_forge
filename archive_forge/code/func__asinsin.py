from typing import Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import devices, ops, protocols
def _asinsin(x: float) -> float:
    """Computes arcsin(sin(x)) for any x. Return value in [-π/2, π/2]."""
    k = round(x / np.pi)
    if k % 2 == 0:
        return x - k * np.pi
    return k * np.pi - x