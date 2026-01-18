import cmath
import math
from typing import AbstractSet, Any, Dict, Optional, Tuple
import numpy as np
import sympy
import cirq
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq.ops import gate_features, raw_types
def _zero_mod_pi(param: 'cirq.TParamVal') -> bool:
    """Returns True iff param, assumed to be in [-pi, pi), is 0 (mod pi)."""
    return param in (0.0, -np.pi, -sympy.pi)