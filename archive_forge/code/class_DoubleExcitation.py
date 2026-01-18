from quantum chemistry applications.
import functools
import numpy as np
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane.operation import Operation
class DoubleExcitation(Operation):
    """
    Double excitation rotation.

    This operation performs an :math:`SO(2)` rotation in the two-dimensional subspace :math:`\\{
    |1100\\rangle,|0011\\rangle\\}`. More precisely, it performs the transformation

    .. math::

        &|0011\\rangle \\rightarrow \\cos(\\phi/2) |0011\\rangle + \\sin(\\phi/2) |1100\\rangle\\\\
        &|1100\\rangle \\rightarrow \\cos(\\phi/2) |1100\\rangle - \\sin(\\phi/2) |0011\\rangle,

    while leaving all other basis states unchanged.

    The name originates from the occupation-number representation of fermionic wavefunctions, where
    the transformation from :math:`|1100\\rangle` to :math:`|0011\\rangle` is interpreted as
    "exciting" two particles from the first pair of qubits to the second pair of qubits.

    **Details:**

    * Number of wires: 4
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: The ``DoubleExcitation`` operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://doi.org/10.1088/1367-2630/ac2cb3):

    Args:
        phi (float): rotation angle :math:`\\phi`
        wires (Sequence[int]): the wires the operation acts on
        id (str or None): String representing the operation (optional)

    **Example**

    The following circuit performs the transformation :math:`|1100\\rangle\\rightarrow \\cos(
    \\phi/2)|1100\\rangle - \\sin(\\phi/2)|0011\\rangle)`:

    .. code-block::

        dev = qml.device('default.qubit', wires=4)

        @qml.qnode(dev)
        def circuit(phi):
            qml.X(0)
            qml.X(1)
            qml.DoubleExcitation(phi, wires=[0, 1, 2, 3])
            return qml.state()

        circuit(0.1)
    """
    num_wires = 4
    'int: Number of wires that the operator acts on.'
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (0,)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    grad_method = 'A'
    'Gradient computation method.'
    parameter_frequencies = [(0.5, 1.0)]
    'Frequencies of the operation parameter with respect to an expectation value.'

    def generator(self):
        w0, w1, w2, w3 = self.wires
        coeffs = [0.0625, 0.0625, -0.0625, 0.0625, -0.0625, 0.0625, -0.0625, -0.0625]
        obs = [qml.X(w0) @ qml.X(w1) @ qml.X(w2) @ qml.Y(w3), qml.X(w0) @ qml.X(w1) @ qml.Y(w2) @ qml.X(w3), qml.X(w0) @ qml.Y(w1) @ qml.X(w2) @ qml.X(w3), qml.X(w0) @ qml.Y(w1) @ qml.Y(w2) @ qml.Y(w3), qml.Y(w0) @ qml.X(w1) @ qml.X(w2) @ qml.X(w3), qml.Y(w0) @ qml.X(w1) @ qml.Y(w2) @ qml.Y(w3), qml.Y(w0) @ qml.Y(w1) @ qml.X(w2) @ qml.Y(w3), qml.Y(w0) @ qml.Y(w1) @ qml.Y(w2) @ qml.X(w3)]
        return qml.Hamiltonian(coeffs, obs)

    def pow(self, z):
        return [DoubleExcitation(self.data[0] * z, wires=self.wires)]

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)
    mask_s = np.zeros((16, 16))
    mask_s[3, 12] = -1
    mask_s[12, 3] = 1

    @staticmethod
    def compute_matrix(phi):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.DoubleExcitation.matrix`


        Args:
          phi (tensor_like or float): rotation angle

        Returns:
          tensor_like: canonical matrix
        """
        return _double_excitations_matrix(phi, 0.0)

    @staticmethod
    def compute_decomposition(phi, wires):
        """Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.DoubleExcitation.decomposition`.

        For the source of this decomposition, see page 17 of
        `"Local, Expressive, Quantum-Number-Preserving VQE Ansatze for Fermionic Systems" <https://doi.org/10.1088/1367-2630/ac2cb3>`_ .

        Args:
            phi (float): rotation angle :math:`\\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.DoubleExcitation.compute_decomposition(1.23, wires=(0,1,2,3))
        [CNOT(wires=[2, 3]),
        CNOT(wires=[0, 2]),
        Hadamard(wires=[3]),
        Hadamard(wires=[0]),
        CNOT(wires=[2, 3]),
        CNOT(wires=[0, 1]),
        RY(0.15375, wires=[1]),
        RY(-0.15375, wires=[0]),
        CNOT(wires=[0, 3]),
        Hadamard(wires=[3]),
        CNOT(wires=[3, 1]),
        RY(0.15375, wires=[1]),
        RY(-0.15375, wires=[0]),
        CNOT(wires=[2, 1]),
        CNOT(wires=[2, 0]),
        RY(-0.15375, wires=[1]),
        RY(0.15375, wires=[0]),
        CNOT(wires=[3, 1]),
        Hadamard(wires=[3]),
        CNOT(wires=[0, 3]),
        RY(-0.15375, wires=[1]),
        RY(0.15375, wires=[0]),
        CNOT(wires=[0, 1]),
        CNOT(wires=[2, 0]),
        Hadamard(wires=[0]),
        Hadamard(wires=[3]),
        CNOT(wires=[0, 2]),
        CNOT(wires=[2, 3])]

        """
        decomp_ops = [qml.CNOT(wires=[wires[2], wires[3]]), qml.CNOT(wires=[wires[0], wires[2]]), qml.Hadamard(wires=wires[3]), qml.Hadamard(wires=wires[0]), qml.CNOT(wires=[wires[2], wires[3]]), qml.CNOT(wires=[wires[0], wires[1]]), qml.RY(phi / 8, wires=wires[1]), qml.RY(-phi / 8, wires=wires[0]), qml.CNOT(wires=[wires[0], wires[3]]), qml.Hadamard(wires=wires[3]), qml.CNOT(wires=[wires[3], wires[1]]), qml.RY(phi / 8, wires=wires[1]), qml.RY(-phi / 8, wires=wires[0]), qml.CNOT(wires=[wires[2], wires[1]]), qml.CNOT(wires=[wires[2], wires[0]]), qml.RY(-phi / 8, wires=wires[1]), qml.RY(phi / 8, wires=wires[0]), qml.CNOT(wires=[wires[3], wires[1]]), qml.Hadamard(wires=wires[3]), qml.CNOT(wires=[wires[0], wires[3]]), qml.RY(-phi / 8, wires=wires[1]), qml.RY(phi / 8, wires=wires[0]), qml.CNOT(wires=[wires[0], wires[1]]), qml.CNOT(wires=[wires[2], wires[0]]), qml.Hadamard(wires=wires[0]), qml.Hadamard(wires=wires[3]), qml.CNOT(wires=[wires[0], wires[2]]), qml.CNOT(wires=[wires[2], wires[3]])]
        return decomp_ops

    def adjoint(self):
        theta, = self.parameters
        return DoubleExcitation(-theta, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or 'G²', cache=cache)