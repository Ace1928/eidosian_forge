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
def _two_qubit_decompose(op):
    """Decomposition for two qubit operations using combination of :class:`~.RZ`, :class:`~.S`,
    :class:`~.Hadamard`, and :class:`~.CNOT`."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        td_ops = qml.ops.two_qubit_decomposition(qml.matrix(op), op.wires)
    d_ops = []
    for td_op in td_ops:
        d_ops.extend(_rot_decompose(td_op) if td_op.num_params and td_op.num_wires == 1 else [td_op])
    return d_ops