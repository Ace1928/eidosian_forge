import functools
import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from .non_parametric_ops import Hadamard, PauliX, PauliY, PauliZ
class U3(Operation):
    """
    Arbitrary single qubit unitary.

    .. math::

        U_3(\\theta, \\phi, \\delta) = \\begin{bmatrix} \\cos(\\theta/2) & -\\exp(i \\delta)\\sin(\\theta/2) \\\\
        \\exp(i \\phi)\\sin(\\theta/2) & \\exp(i (\\phi + \\delta))\\cos(\\theta/2) \\end{bmatrix}

    The :math:`U_3` gate is related to the single-qubit rotation :math:`R` (:class:`Rot`) and the
    :math:`R_\\phi` (:class:`PhaseShift`) gates via the following relation:

    .. math::

        U_3(\\theta, \\phi, \\delta) = R_\\phi(\\phi+\\delta) R(\\delta,\\theta,-\\delta)

    .. note::

        If the ``U3`` gate is not supported on the targeted device, PennyLane
        will attempt to decompose the gate into :class:`~.PhaseShift` and :class:`~.Rot` gates.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 3
    * Number of dimensions per parameter: (0, 0, 0)
    * Gradient recipe: :math:`\\frac{d}{d\\phi}f(U_3(\\theta, \\phi, \\delta)) = \\frac{1}{2}\\left[f(U_3(\\theta+\\pi/2, \\phi, \\delta)) - f(U_3(\\theta-\\pi/2, \\phi, \\delta))\\right]`
      where :math:`f` is an expectation value depending on :math:`U_3(\\theta, \\phi, \\delta)`.
      This gradient recipe applies for each angle argument :math:`\\{\\theta, \\phi, \\delta\\}`.

    Args:
        theta (float): polar angle :math:`\\theta`
        phi (float): azimuthal angle :math:`\\phi`
        delta (float): quantum phase :math:`\\delta`
        wires (Sequence[int] or int): the subsystem the gate acts on
        id (str or None): String representing the operation (optional)
    """
    num_wires = 1
    num_params = 3
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (0, 0, 0)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    grad_method = 'A'
    parameter_frequencies = [(1,), (1,), (1,)]

    def __init__(self, theta, phi, delta, wires, id=None):
        super().__init__(theta, phi, delta, wires=wires, id=id)

    @staticmethod
    def compute_matrix(theta, phi, delta):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.U3.matrix`

        Args:
            theta (tensor_like or float): polar angle
            phi (tensor_like or float): azimuthal angle
            delta (tensor_like or float): quantum phase

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.U3.compute_matrix(torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3))
        tensor([[ 0.9988+0.0000j, -0.0477-0.0148j],
                [ 0.0490+0.0099j,  0.8765+0.4788j]])

        """
        interface = qml.math.get_interface(theta, phi, delta)
        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)
        if interface == 'tensorflow':
            phi = qml.math.cast_like(qml.math.asarray(phi, like=interface), 1j)
            delta = qml.math.cast_like(qml.math.asarray(delta, like=interface), 1j)
            c = qml.math.cast_like(qml.math.asarray(c, like=interface), 1j)
            s = qml.math.cast_like(qml.math.asarray(s, like=interface), 1j)
        one = qml.math.ones_like(phi) * qml.math.ones_like(delta)
        c = c * one
        s = s * one
        mat = [[c, -s * qml.math.exp(1j * delta)], [s * qml.math.exp(1j * phi), c * qml.math.exp(1j * (phi + delta))]]
        return qml.math.stack([stack_last(row) for row in mat], axis=-2)

    @staticmethod
    def compute_decomposition(theta, phi, delta, wires):
        """Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.U3.decomposition`.

        Args:
            theta (float): polar angle :math:`\\theta`
            phi (float): azimuthal angle :math:`\\phi`
            delta (float): quantum phase :math:`\\delta`
            wires (Iterable, Wires): the subsystem the gate acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.U3.compute_decomposition(1.23, 2.34, 3.45, wires=0)
        [Rot(3.45, 1.23, -3.45, wires=[0]),
        PhaseShift(3.45, wires=[0]),
        PhaseShift(2.34, wires=[0])]

        """
        return [Rot(delta, theta, -delta, wires=wires), PhaseShift(delta, wires=wires), PhaseShift(phi, wires=wires)]

    def adjoint(self):
        theta, phi, delta = self.parameters
        new_delta = qml.math.mod(np.pi - phi, 2 * np.pi)
        new_phi = qml.math.mod(np.pi - delta, 2 * np.pi)
        return U3(theta, new_phi, new_delta, wires=self.wires)

    def simplify(self):
        """Simplifies into :class:`~.RX`, :class:`~.RY`, or :class:`~.PhaseShift` gates
        if possible.

        >>> qml.U3(0.1, 0, 0, wires=0).simplify()
        RY(0.1, wires=[0])

        """
        wires = self.wires
        params = self.parameters
        p0 = params[0] % (4 * np.pi)
        p1, p2 = [p % (2 * np.pi) for p in params[1:]]
        if _can_replace(p0, 0) and _can_replace(p1, 0) and _can_replace(p2, 0):
            return qml.Identity(wires=wires)
        if _can_replace(p0, 0) and (not _can_replace(p1, 0)) and _can_replace(p2, 0):
            return PhaseShift(p1, wires=wires)
        if _can_replace(p2, np.pi / 2) and _can_replace(p1, 3 * np.pi / 2) and (not _can_replace(p0, 0)):
            return RX(p0, wires=wires)
        if not _can_replace(p0, 0) and _can_replace(p1, 0) and _can_replace(p2, 0):
            return RY(p0, wires=wires)
        return U3(p0, p1, p2, wires=wires)