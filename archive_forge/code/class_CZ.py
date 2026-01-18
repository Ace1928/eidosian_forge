import warnings
from typing import Iterable
from functools import lru_cache
import numpy as np
from scipy.linalg import block_diag
import pennylane as qml
from pennylane.operation import (
from pennylane.ops.qubit.matrix_ops import QubitUnitary
from pennylane.ops.qubit.parametric_ops_single_qubit import stack_last
from .controlled import ControlledOp
from .controlled_decompositions import decompose_mcx
class CZ(ControlledOp):
    """CZ(wires)
    The controlled-Z operator

    .. math:: CZ = \\begin{bmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0\\\\
            0 & 0 & 1 & 0\\\\
            0 & 0 & 0 & -1
        \\end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """
    num_wires = 2
    'int: Number of wires that the operator acts on.'
    num_params = 0
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = ()
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    name = 'CZ'

    def _flatten(self):
        return (tuple(), (self.wires,))

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(metadata[0])

    def __init__(self, wires, id=None):
        control_wire, wire = wires
        super().__init__(qml.Z(wires=wire), control_wire, id=id)

    def __repr__(self):
        return f'CZ(wires={self.wires.tolist()})'

    @staticmethod
    @lru_cache()
    def compute_matrix():
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CZ.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.CZ.compute_matrix())
        [[ 1  0  0  0]
         [ 0  1  0  0]
         [ 0  0  1  0]
         [ 0  0  0 -1]]
        """
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

    def _controlled(self, wire):
        return qml.CCZ(wires=wire + self.wires)

    @staticmethod
    def compute_decomposition(wires):
        return [qml.ControlledPhaseShift(np.pi, wires=wires)]