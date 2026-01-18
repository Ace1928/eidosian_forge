from __future__ import annotations
import copy
from abc import ABC
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.op_shape import OpShape
from .mixins import GroupMixin
@property
def _input_dim(self):
    """Return the total input dimension."""
    return self._op_shape._dim_r