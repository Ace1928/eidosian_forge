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
class CH(ControlledOp):
    """CH(wires)
    The controlled-Hadamard operator

    .. math:: CH = \\begin{bmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & \\frac{1}{\\sqrt{2}} & \\frac{1}{\\sqrt{2}} \\\\
            0 & 0 & \\frac{1}{\\sqrt{2}} & -\\frac{1}{\\sqrt{2}}
        \\end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """
    num_wires = 2
    'int: Number of wires that the operation acts on.'
    num_params = 0
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = ()
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    name = 'CH'

    def _flatten(self):
        return (tuple(), (self.wires,))

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(metadata[0])

    def __init__(self, wires, id=None):
        control_wires = wires[:1]
        target_wires = wires[1:]
        super().__init__(qml.Hadamard(wires=target_wires), control_wires, id=id)

    def __repr__(self):
        return f'CH(wires={self.wires.tolist()})'

    @staticmethod
    @lru_cache()
    def compute_matrix():
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CH.matrix`


        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.CH.compute_matrix())
        [[ 1.          0.          0.          0.        ]
         [ 0.          1.          0.          0.        ]
         [ 0.          0.          0.70710678  0.70710678]
         [ 0.          0.          0.70710678 -0.70710678]]
        """
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, INV_SQRT2, INV_SQRT2], [0, 0, INV_SQRT2, -INV_SQRT2]])

    @staticmethod
    def compute_decomposition(wires):
        """Representation of the operator as a product of other operators (static method).


        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.CH.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.CH.compute_decomposition([0, 1]))
        [RY(-0.7853981633974483, wires=[1]), CZ(wires=[0, 1]), RY(0.7853981633974483, wires=[1])]

        """
        return [qml.RY(-np.pi / 4, wires=wires[1]), qml.CZ(wires=wires), qml.RY(+np.pi / 4, wires=wires[1])]