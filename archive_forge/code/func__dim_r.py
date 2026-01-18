from __future__ import annotations
import copy
from functools import reduce
from operator import mul
from math import log2
from numbers import Integral
from qiskit.exceptions import QiskitError
@property
def _dim_r(self):
    """Return the total input dimension."""
    if self._dims_r:
        return reduce(mul, self._dims_r)
    return 2 ** self._num_qargs_r