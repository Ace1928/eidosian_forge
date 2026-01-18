import math
from typing import List, Optional, Tuple
import numpy as np
import sympy
from cirq import ops, linalg, protocols
from cirq.linalg.tolerance import near_zero_mod
def _signed_mod_1(x: float) -> float:
    return (x + 0.5) % 1 - 0.5