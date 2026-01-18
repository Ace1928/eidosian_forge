import itertools
import uuid
from typing import Iterable
from qiskit.circuit import (
from qiskit.circuit.classical import expr
def _for_loop_eq(node1, node2, bit_indices1, bit_indices2):
    indexset1, param1, body1 = node1.op.params
    indexset2, param2, body2 = node2.op.params
    if indexset1 != indexset2:
        return False
    if param1 is None and param2 is not None or (param1 is not None and param2 is None):
        return False
    if param1 is not None and param2 is not None:
        sentinel = Parameter(str(uuid.uuid4()))
        body1 = body1.assign_parameters({param1: sentinel}, inplace=False) if param1 in body1.parameters else body1
        body2 = body2.assign_parameters({param2: sentinel}, inplace=False) if param2 in body2.parameters else body2
    return _circuit_to_dag(body1, node1.qargs, node1.cargs, bit_indices1) == _circuit_to_dag(body2, node2.qargs, node2.cargs, bit_indices2)