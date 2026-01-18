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
class CRX(ControlledOp):
    """The controlled-RX operator

    .. math::

        \\begin{align}
            CR_x(\\phi) &=
            \\begin{bmatrix}
            & 1 & 0 & 0 & 0 \\\\
            & 0 & 1 & 0 & 0\\\\
            & 0 & 0 & \\cos(\\phi/2) & -i\\sin(\\phi/2)\\\\
            & 0 & 0 & -i\\sin(\\phi/2) & \\cos(\\phi/2)
            \\end{bmatrix}.
        \\end{align}

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: The controlled-RX operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://doi.org/10.1088/1367-2630/ac2cb3):

      .. math::

          \\frac{d}{d\\phi}f(CR_x(\\phi)) = c_+ \\left[f(CR_x(\\phi+a)) - f(CR_x(\\phi-a))\\right] - c_- \\left[f(CR_x(\\phi+b)) - f(CR_x(\\phi-b))\\right]

      where :math:`f` is an expectation value depending on :math:`CR_x(\\phi)`, and

      - :math:`a = \\pi/2`
      - :math:`b = 3\\pi/2`
      - :math:`c_{\\pm} = (\\sqrt{2} \\pm 1)/{4\\sqrt{2}}`

    Args:
        phi (float): rotation angle :math:`\\phi`
        wires (Sequence[int]): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """
    num_wires = 2
    'int: Number of wires that the operation acts on.'
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (0,)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    name = 'CRX'
    parameter_frequencies = [(0.5, 1.0)]

    def __init__(self, phi, wires, id=None):
        super().__init__(qml.RX(phi, wires=wires[1:]), control_wires=wires[0], id=id)

    def __repr__(self):
        return f'CRX({self.data[0]}, wires={self.wires.tolist()})'

    @classmethod
    def _unflatten(cls, data, metadata):
        base = data[0]
        control_wires = metadata[0]
        return cls(*base.data, wires=control_wires + base.wires)

    @staticmethod
    def compute_matrix(theta):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CRX.matrix`

        Args:
            theta (tensor_like or float): rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.CRX.compute_matrix(torch.tensor(0.5))
        tensor([[1.0+0.0j, 0.0+0.0j,    0.0+0.0j,    0.0+0.0j],
                [0.0+0.0j, 1.0+0.0j,    0.0+0.0j,    0.0+0.0j],
                [0.0+0.0j, 0.0+0.0j, 0.9689+0.0j, 0.0-0.2474j],
                [0.0+0.0j, 0.0+0.0j, 0.0-0.2474j, 0.9689+0.0j]])
        """
        interface = qml.math.get_interface(theta)
        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)
        if interface == 'tensorflow':
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)
        c = (1 + 0j) * c
        js = -1j * s
        ones = qml.math.ones_like(js)
        zeros = qml.math.zeros_like(js)
        matrix = [[ones, zeros, zeros, zeros], [zeros, ones, zeros, zeros], [zeros, zeros, c, js], [zeros, zeros, js, c]]
        return qml.math.stack([stack_last(row) for row in matrix], axis=-2)

    @staticmethod
    def compute_decomposition(phi, wires):
        """Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.CRot.decomposition`.

        Args:
            phi (float): rotation angle :math:`\\phi`
            wires (Iterable, Wires): the wires the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.CRX.compute_decomposition(1.2, wires=(0,1))
        [RZ(1.5707963267948966, wires=[1]),
        RY(0.6, wires=[1]),
        CNOT(wires=[0, 1]),
        RY(-0.6, wires=[1]),
        CNOT(wires=[0, 1]),
        RZ(-1.5707963267948966, wires=[1])]

        """
        pi_half = qml.math.ones_like(phi) * (np.pi / 2)
        return [qml.RZ(pi_half, wires=wires[1]), qml.RY(phi / 2, wires=wires[1]), qml.CNOT(wires=wires), qml.RY(-phi / 2, wires=wires[1]), qml.CNOT(wires=wires), qml.RZ(-pi_half, wires=wires[1])]