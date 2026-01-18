import functools
import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from .non_parametric_ops import Hadamard, PauliX, PauliY, PauliZ
class Rot(Operation):
    """
    Arbitrary single qubit rotation

    .. math::

        R(\\phi,\\theta,\\omega) = RZ(\\omega)RY(\\theta)RZ(\\phi)= \\begin{bmatrix}
        e^{-i(\\phi+\\omega)/2}\\cos(\\theta/2) & -e^{i(\\phi-\\omega)/2}\\sin(\\theta/2) \\\\
        e^{-i(\\phi-\\omega)/2}\\sin(\\theta/2) & e^{i(\\phi+\\omega)/2}\\cos(\\theta/2)
        \\end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 3
    * Number of dimensions per parameter: (0, 0, 0)
    * Gradient recipe: :math:`\\frac{d}{d\\phi}f(R(\\phi, \\theta, \\omega)) = \\frac{1}{2}\\left[f(R(\\phi+\\pi/2, \\theta, \\omega)) - f(R(\\phi-\\pi/2, \\theta, \\omega))\\right]`
      where :math:`f` is an expectation value depending on :math:`R(\\phi, \\theta, \\omega)`.
      This gradient recipe applies for each angle argument :math:`\\{\\phi, \\theta, \\omega\\}`.

    .. note::

        If the ``Rot`` gate is not supported on the targeted device, PennyLane
        will attempt to decompose the gate into :class:`~.RZ` and :class:`~.RY` gates.

    Args:
        phi (float): rotation angle :math:`\\phi`
        theta (float): rotation angle :math:`\\theta`
        omega (float): rotation angle :math:`\\omega`
        wires (Any, Wires): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """
    num_wires = 1
    num_params = 3
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (0, 0, 0)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    grad_method = 'A'
    parameter_frequencies = [(1,), (1,), (1,)]

    def __init__(self, phi, theta, omega, wires, id=None):
        super().__init__(phi, theta, omega, wires=wires, id=id)

    @staticmethod
    def compute_matrix(phi, theta, omega):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Rot.matrix`


        Args:
            phi (tensor_like or float): first rotation angle
            theta (tensor_like or float): second rotation angle
            omega (tensor_like or float): third rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.Rot.compute_matrix(torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3))
        tensor([[ 0.9752-0.1977j, -0.0993+0.0100j],
                [ 0.0993+0.0100j,  0.9752+0.1977j]])

        """
        interface = qml.math.get_interface(phi, theta, omega)
        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)
        if interface == 'tensorflow':
            phi = qml.math.cast_like(qml.math.asarray(phi, like=interface), 1j)
            omega = qml.math.cast_like(qml.math.asarray(omega, like=interface), 1j)
            c = qml.math.cast_like(qml.math.asarray(c, like=interface), 1j)
            s = qml.math.cast_like(qml.math.asarray(s, like=interface), 1j)
        one = qml.math.ones_like(phi) * qml.math.ones_like(omega)
        c = c * one
        s = s * one
        mat = [[qml.math.exp(-0.5j * (phi + omega)) * c, -qml.math.exp(0.5j * (phi - omega)) * s], [qml.math.exp(-0.5j * (phi - omega)) * s, qml.math.exp(0.5j * (phi + omega)) * c]]
        return qml.math.stack([stack_last(row) for row in mat], axis=-2)

    @staticmethod
    def compute_decomposition(phi, theta, omega, wires):
        """Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.Rot.decomposition`.

        Args:
            phi (float): rotation angle :math:`\\phi`
            theta (float): rotation angle :math:`\\theta`
            omega (float): rotation angle :math:`\\omega`
            wires (Any, Wires): the wire the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.Rot.compute_decomposition(1.2, 2.3, 3.4, wires=0)
        [RZ(1.2, wires=[0]), RY(2.3, wires=[0]), RZ(3.4, wires=[0])]

        """
        return [RZ(phi, wires=wires), RY(theta, wires=wires), RZ(omega, wires=wires)]

    def adjoint(self):
        phi, theta, omega = self.parameters
        return Rot(-omega, -theta, -phi, wires=self.wires)

    def _controlled(self, wire):
        return qml.CRot(*self.parameters, wires=wire + self.wires)

    def single_qubit_rot_angles(self):
        return self.data

    def simplify(self):
        """Simplifies into single-rotation gates or a Hadamard if possible.

        >>> qml.Rot(np.pi / 2, 0.1, -np.pi / 2, wires=0).simplify()
        RX(0.1, wires=[0])
        >>> qml.Rot(np.pi, np.pi/2, 0, 0).simplify()
        Hadamard(wires=[0])

        """
        p0, p1, p2 = [p % (4 * np.pi) for p in self.data]
        if _can_replace(p0, 0) and _can_replace(p1, 0) and _can_replace(p2, 0):
            return qml.Identity(wires=self.wires)
        if _can_replace(p0, np.pi / 2) and _can_replace(p2, 7 * np.pi / 2):
            return RX(p1, wires=self.wires)
        if _can_replace(p0, 0) and _can_replace(p2, 0):
            return RY(p1, wires=self.wires)
        if _can_replace(p1, 0):
            return RZ((p0 + p2) % (4 * np.pi), wires=self.wires)
        if _can_replace(p0, np.pi) and _can_replace(p1, np.pi / 2) and _can_replace(p2, 0):
            return Hadamard(wires=self.wires)
        return Rot(p0, p1, p2, wires=self.wires)