from typing import Any
import numpy as np
import sympy
from cirq import protocols, linalg
from cirq.testing import lin_alg_utils
Uses `val._unitary_` to check `val._phase_by_`'s behavior.