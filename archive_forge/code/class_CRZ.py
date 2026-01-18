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
class CRZ(ControlledOp):
    """The controlled-RZ operator

    .. math::

        \\begin{align}
             CR_z(\\phi) &=
             \\begin{bmatrix}
                1 & 0 & 0 & 0 \\\\
                0 & 1 & 0 & 0\\\\
                0 & 0 & e^{-i\\phi/2} & 0\\\\
                0 & 0 & 0 & e^{i\\phi/2}
            \\end{bmatrix}.
        \\end{align}


    .. note:: The subscripts of the operations in the formula refer to the wires they act on, e.g. 1 corresponds
        to the first element in ``wires`` that is the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: The controlled-RZ operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://doi.org/10.1088/1367-2630/ac2cb3):

      .. math::

          \\frac{d}{d\\phi}f(CR_z(\\phi)) = c_+ \\left[f(CR_z(\\phi+a)) - f(CR_z(\\phi-a))\\right] - c_- \\left[f(CR_z(\\phi+b)) - f(CR_z(\\phi-b))\\right]

      where :math:`f` is an expectation value depending on :math:`CR_z(\\phi)`, and

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
    name = 'CRZ'
    parameter_frequencies = [(0.5, 1.0)]

    def __init__(self, phi, wires, id=None):
        super().__init__(qml.RZ(phi, wires=wires[1:]), control_wires=wires[0], id=id)

    def __repr__(self):
        return f'CRZ({self.data[0]}, wires={self.wires})'

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

        .. seealso:: :meth:`~.CRZ.matrix`

        Args:
            theta (tensor_like or float): rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.CRZ.compute_matrix(torch.tensor(0.5))
        tensor([[1.0+0.0j, 0.0+0.0j,       0.0+0.0j,       0.0+0.0j],
                [0.0+0.0j, 1.0+0.0j,       0.0+0.0j,       0.0+0.0j],
                [0.0+0.0j, 0.0+0.0j, 0.9689-0.2474j,       0.0+0.0j],
                [0.0+0.0j, 0.0+0.0j,       0.0+0.0j, 0.9689+0.2474j]])
        """
        if qml.math.get_interface(theta) == 'tensorflow':
            p = qml.math.exp(-0.5j * qml.math.cast_like(theta, 1j))
            if qml.math.ndim(p) == 0:
                return qml.math.diag([1, 1, p, qml.math.conj(p)])
            ones = qml.math.ones_like(p)
            diags = stack_last([ones, ones, p, qml.math.conj(p)])
            return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(4, like=diags), diags)
        signs = qml.math.array([0, 0, 1, -1], like=theta)
        arg = -0.5j * theta
        if qml.math.ndim(arg) == 0:
            return qml.math.diag(qml.math.exp(arg * signs))
        diags = qml.math.exp(qml.math.outer(arg, signs))
        return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(4, like=diags), diags)

    @staticmethod
    def compute_eigvals(theta, **_):
        """Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \\Sigma U^{\\dagger},

        where :math:`\\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.CRZ.eigvals`


        Args:
            theta (tensor_like or float): rotation angle

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.CRZ.compute_eigvals(torch.tensor(0.5))
        tensor([1.0000+0.0000j, 1.0000+0.0000j, 0.9689-0.2474j, 0.9689+0.2474j])
        """
        if qml.math.get_interface(theta) == 'tensorflow':
            phase = qml.math.exp(-0.5j * qml.math.cast_like(theta, 1j))
            ones = qml.math.ones_like(phase)
            return stack_last([ones, ones, phase, qml.math.conj(phase)])
        prefactors = qml.math.array([0, 0, -0.5j, 0.5j], like=theta)
        if qml.math.ndim(theta) == 0:
            product = theta * prefactors
        else:
            product = qml.math.outer(theta, prefactors)
        return qml.math.exp(product)

    def eigvals(self):
        return self.compute_eigvals(*self.parameters)

    @staticmethod
    def compute_decomposition(phi, wires):
        """Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.CRZ.decomposition`.

        Args:
            phi (float): rotation angle :math:`\\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.CRZ.compute_decomposition(1.2, wires=(0,1))
        [PhaseShift(0.6, wires=[1]),
        CNOT(wires=[0, 1]),
        PhaseShift(-0.6, wires=[1]),
        CNOT(wires=[0, 1])]

        """
        return [qml.PhaseShift(phi / 2, wires=wires[1]), qml.CNOT(wires=wires), qml.PhaseShift(-phi / 2, wires=wires[1]), qml.CNOT(wires=wires)]