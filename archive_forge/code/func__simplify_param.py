import math
import warnings
from itertools import product
from typing import Sequence, Callable
import pennylane as qml
from pennylane.ops import Adjoint
from pennylane.queuing import QueuingManager
from pennylane.transforms.core import transform
from pennylane.tape import QuantumTape
from pennylane.transforms.optimization import (
from pennylane.transforms.optimization.optimization_utils import find_next_gate, _fuse_global_phases
from pennylane.ops.op_math.decompositions.solovay_kitaev import sk_decomposition
def _simplify_param(theta, gate):
    """Check if the parameter allows simplification for the rotation gate.

    For the cases where theta is an integer multiple of π: (a) returns a global phase
    when even, and (b) returns combination of provided gate with global phase when odd.
    In rest of the other cases it would return None.
    """
    if qml.math.is_abstract(theta):
        return None
    if qml.math.allclose(theta, 0.0, atol=1e-06):
        return [qml.GlobalPhase(0.0)]
    rem_, mod_ = (qml.math.divide(theta, math.pi), qml.math.mod(theta, math.pi))
    if qml.math.allclose(mod_, 0.0, atol=1e-06):
        ops = [qml.GlobalPhase(theta / 2)]
        if qml.math.allclose(qml.math.mod(rem_, 2), 1.0, atol=1e-06):
            ops.append(gate)
        return ops
    return None