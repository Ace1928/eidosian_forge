from __future__ import annotations
import copy
from functools import reduce
from operator import mul
from math import log2
from numbers import Integral
from qiskit.exceptions import QiskitError
def _validate_add(self, other, qargs=None):
    if qargs:
        if self._num_qargs_l != self._num_qargs_r:
            raise QiskitError('Cannot add using qargs if number of left and right qargs are not equal.')
        if self.dims_l(qargs) != other.dims_l():
            raise QiskitError('Cannot add shapes width different left dimension on specified qargs {} != {}'.format(self.dims_l(qargs), other.dims_l()))
        if self.dims_r(qargs) != other.dims_r():
            raise QiskitError('Cannot add shapes width different total right dimension on specified qargs{} != {}'.format(self.dims_r(qargs), other.dims_r()))
    elif self != other:
        if self._dim_l != other._dim_l:
            raise QiskitError('Cannot add shapes width different total left dimension {} != {}'.format(self._dim_l, other._dim_l))
        if self._dim_r != other._dim_r:
            raise QiskitError('Cannot add shapes width different total right dimension {} != {}'.format(self._dim_r, other._dim_r))
    return self