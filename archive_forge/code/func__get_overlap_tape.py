import warnings
from scipy.linalg import sqrtm
import numpy as np
import pennylane as qml
def _get_overlap_tape(self, cost, args1, args2, kwargs):
    op_forward = self._get_operations(cost, args1, kwargs)
    op_inv = self._get_operations(cost, args2, kwargs)
    new_ops = op_forward + [qml.adjoint(op) for op in reversed(op_inv)]
    return qml.tape.QuantumScript(new_ops, [qml.probs(wires=cost.tape.wires.labels)])