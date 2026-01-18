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
class ControlledPhaseShift(ControlledOp):
    """A qubit controlled phase shift.

    .. math:: CR_\\phi(\\phi) = \\begin{bmatrix}
                1 & 0 & 0 & 0 \\\\
                0 & 1 & 0 & 0 \\\\
                0 & 0 & 1 & 0 \\\\
                0 & 0 & 0 & e^{i\\phi}
            \\end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\\frac{d}{d\\phi}f(CR_\\phi(\\phi)) = \\frac{1}{2}\\left[f(CR_\\phi(\\phi+\\pi/2)) - f(CR_\\phi(\\phi-\\pi/2))\\right]`
        where :math:`f` is an expectation value depending on :math:`CR_{\\phi}(\\phi)`.

    Args:
        phi (float): rotation angle :math:`\\phi`
        wires (Sequence[int]): the wire the operation acts on
        id (str or None): String representing the operation (optional)

    """
    num_wires = 2
    'int: Number of wires the operator acts on.'
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (0,)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    name = 'ControlledPhaseShift'
    parameter_frequencies = [(1,)]

    def __init__(self, phi, wires, id=None):
        super().__init__(qml.PhaseShift(phi, wires=wires[1:]), control_wires=wires[0], id=id)

    def __repr__(self):
        return f'ControlledPhaseShift({self.data[0]}, wires={self.wires})'

    @classmethod
    def _unflatten(cls, data, metadata):
        base = data[0]
        control_wires = metadata[0]
        return cls(*base.data, wires=control_wires + base.wires)

    @staticmethod
    def compute_matrix(phi):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.ControlledPhaseShift.matrix`

        Args:
            phi (tensor_like or float): phase shift

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.ControlledPhaseShift.compute_matrix(torch.tensor(0.5))
            tensor([[1.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0000+0.0000j],
                    [0.0+0.0j, 1.0+0.0j, 0.0+0.0j, 0.0000+0.0000j],
                    [0.0+0.0j, 0.0+0.0j, 1.0+0.0j, 0.0000+0.0000j],
                    [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.8776+0.4794j]])
        """
        if qml.math.get_interface(phi) == 'tensorflow':
            p = qml.math.exp(1j * qml.math.cast_like(phi, 1j))
            if qml.math.ndim(p) == 0:
                return qml.math.diag([1, 1, 1, p])
            ones = qml.math.ones_like(p)
            diags = stack_last([ones, ones, ones, p])
            return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(4, like=diags), diags)
        signs = qml.math.array([0, 0, 0, 1], like=phi)
        arg = 1j * phi
        if qml.math.ndim(arg) == 0:
            return qml.math.diag(qml.math.exp(arg * signs))
        diags = qml.math.exp(qml.math.outer(arg, signs))
        return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(4, like=diags), diags)

    @staticmethod
    def compute_eigvals(phi, **_):
        """Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \\Sigma U^{\\dagger},

        where :math:`\\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.ControlledPhaseShift.eigvals`


        Args:
            phi (tensor_like or float): phase shift

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.ControlledPhaseShift.compute_eigvals(torch.tensor(0.5))
        tensor([1.0000+0.0000j, 1.0000+0.0000j, 1.0000+0.0000j, 0.8776+0.4794j])
        """
        if qml.math.get_interface(phi) == 'tensorflow':
            phase = qml.math.exp(1j * qml.math.cast_like(phi, 1j))
            ones = qml.math.ones_like(phase)
            return stack_last([ones, ones, ones, phase])
        prefactors = qml.math.array([0, 0, 0, 1j], like=phi)
        if qml.math.ndim(phi) == 0:
            product = phi * prefactors
        else:
            product = qml.math.outer(phi, prefactors)
        return qml.math.exp(product)

    def eigvals(self):
        return self.compute_eigvals(*self.parameters)

    @staticmethod
    def compute_decomposition(phi, wires):
        """Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \\dots O_n.

        .. seealso:: :meth:`~.ControlledPhaseShift.decomposition`.

        Args:
            phi (float): rotation angle :math:`\\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.ControlledPhaseShift.compute_decomposition(1.234, wires=(0,1))
        [PhaseShift(0.617, wires=[0]),
         CNOT(wires=[0, 1]),
         PhaseShift(-0.617, wires=[1]),
         CNOT(wires=[0, 1]),
         PhaseShift(0.617, wires=[1])]

        """
        return [qml.PhaseShift(phi / 2, wires=wires[0]), qml.CNOT(wires=wires), qml.PhaseShift(-phi / 2, wires=wires[1]), qml.CNOT(wires=wires), qml.PhaseShift(phi / 2, wires=wires[1])]