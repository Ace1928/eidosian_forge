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
class Toffoli(ControlledOp):
    """Toffoli(wires)
    Toffoli (controlled-controlled-X) gate.

    .. math::

        Toffoli =
        \\begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\\\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
        \\end{pmatrix}

    **Details:**

    * Number of wires: 3
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the subsystem the gate acts on
    """
    num_wires = 3
    'int: Number of wires that the operator acts on.'
    num_params = 0
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = ()
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    arithmetic_depth = 0
    name = 'Toffoli'

    def _flatten(self):
        return (tuple(), (self.wires,))

    @classmethod
    def _unflatten(cls, _, metadata):
        return cls(metadata[0])

    def __init__(self, wires, id=None):
        control_wires = wires[:2]
        target_wires = wires[2:]
        super().__init__(qml.PauliX(wires=target_wires), control_wires, id=id)

    def __repr__(self):
        return f'Toffoli(wires={self.wires.tolist()})'

    @staticmethod
    @lru_cache()
    def compute_matrix():
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Toffoli.matrix`


        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.Toffoli.compute_matrix())
        [[1 0 0 0 0 0 0 0]
         [0 1 0 0 0 0 0 0]
         [0 0 1 0 0 0 0 0]
         [0 0 0 1 0 0 0 0]
         [0 0 0 0 1 0 0 0]
         [0 0 0 0 0 1 0 0]
         [0 0 0 0 0 0 0 1]
         [0 0 0 0 0 0 1 0]]
        """
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0]])

    @staticmethod
    def compute_decomposition(wires):
        """Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.Toffoli.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.Toffoli.compute_decomposition((0,1,2))
        [Hadamard(wires=[2]),
         CNOT(wires=[1, 2]),
         Adjoint(T(wires=[2])),
         CNOT(wires=[0, 2]),
         T(wires=[2]),
         CNOT(wires=[1, 2]),
         Adjoint(T(wires=[2])),
         CNOT(wires=[0, 2]),
         T(wires=[2]),
         T(wires=[1]),
         CNOT(wires=[0, 1]),
         Hadamard(wires=[2]),
         T(wires=[0]),
         Adjoint(T(wires=[1])),
         CNOT(wires=[0, 1])]

        """
        return [qml.Hadamard(wires=wires[2]), CNOT(wires=[wires[1], wires[2]]), qml.adjoint(qml.T(wires=wires[2])), CNOT(wires=[wires[0], wires[2]]), qml.T(wires=wires[2]), CNOT(wires=[wires[1], wires[2]]), qml.adjoint(qml.T(wires=wires[2])), CNOT(wires=[wires[0], wires[2]]), qml.T(wires=wires[2]), qml.T(wires=wires[1]), CNOT(wires=[wires[0], wires[1]]), qml.Hadamard(wires=wires[2]), qml.T(wires=wires[0]), qml.adjoint(qml.T(wires=wires[1])), CNOT(wires=[wires[0], wires[1]])]