import functools
from operator import matmul
import numpy as np
import pennylane as qml
from pennylane.math import expand_matrix
from pennylane.operation import AnyWires, Operation
from pennylane.utils import pauli_eigs
from pennylane.wires import Wires
from .non_parametric_ops import Hadamard, PauliX, PauliY, PauliZ
from .parametric_ops_single_qubit import _can_replace, stack_last, RX, RY, RZ, PhaseShift
class IsingXY(Operation):
    """
    Ising (XX + YY) coupling gate

    .. math:: \\mathtt{XY}(\\phi) = \\exp(i \\frac{\\theta}{4} (X \\otimes X + Y \\otimes Y)) =
        \\begin{bmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & \\cos(\\phi / 2) & i \\sin(\\phi / 2) & 0 \\\\
            0 & i \\sin(\\phi / 2) & \\cos(\\phi / 2) & 0 \\\\
            0 & 0 & 0 & 1
        \\end{bmatrix}.

    .. note::

        Special cases of using the :math:`XY` operator include:

        * :math:`XY(0) = I`;
        * :math:`XY(\\frac{\\pi}{2}) = \\sqrt{iSWAP}`;
        * :math:`XY(\\pi) = iSWAP`;

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: The XY operator satisfies a four-term parameter-shift rule

      .. math::
          \\frac{d}{d \\phi} f(XY(\\phi))
          = c_+ \\left[ f(XY(\\phi + a)) - f(XY(\\phi - a)) \\right]
          - c_- \\left[ f(XY(\\phi + b)) - f(XY(\\phi - b)) \\right]

      where :math:`f` is an expectation value depending on :math:`XY(\\phi)`, and

      - :math:`a = \\pi / 2`
      - :math:`b = 3 \\pi / 2`
      - :math:`c_{\\pm} = (\\sqrt{2} \\pm 1)/{4 \\sqrt{2}}`

    Args:
        phi (float): the phase angle
        wires (int): the subsystem the gate acts on
        id (str or None): String representing the operation (optional)
    """
    num_wires = 2
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (0,)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    grad_method = 'A'
    parameter_frequencies = [(0.5, 1.0)]

    def generator(self):
        return 0.25 * (PauliX(wires=self.wires[0]) @ PauliX(wires=self.wires[1]) + PauliY(wires=self.wires[0]) @ PauliY(wires=self.wires[1]))

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(phi, wires):
        """Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.IsingXY.decomposition`.

        Args:
            phi (float): the phase angle
            wires (Iterable, Wires): the subsystem the gate acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.IsingXY.compute_decomposition(1.23, wires=(0,1))
        [Hadamard(wires=[0]), CY(wires=[0, 1]), RY(0.615, wires=[0]), RX(-0.615, wires=[1]), CY(wires=[0, 1]), Hadamard(wires=[0])]

        """
        return [Hadamard(wires=[wires[0]]), qml.CY(wires=wires), RY(phi / 2, wires=[wires[0]]), RX(-phi / 2, wires=[wires[1]]), qml.CY(wires=wires), Hadamard(wires=[wires[0]])]

    @staticmethod
    def compute_matrix(phi):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.IsingXY.matrix`


        Args:
           phi (tensor_like or float): phase angle

        Returns:
           tensor_like: canonical matrix

        **Example**

        >>> qml.IsingXY.compute_matrix(0.5)
        array([[1.        +0.j        , 0.        +0.j        ,        0.        +0.j        , 0.        +0.j        ],
               [0.        +0.j        , 0.96891242+0.j        ,        0.        +0.24740396j, 0.        +0.j        ],
               [0.        +0.j        , 0.        +0.24740396j,        0.96891242+0.j        , 0.        +0.j        ],
               [0.        +0.j        , 0.        +0.j        ,        0.        +0.j        , 1.        +0.j        ]])
        """
        c = qml.math.cos(phi / 2)
        s = qml.math.sin(phi / 2)
        if qml.math.get_interface(phi) == 'tensorflow':
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)
        js = 1j * s
        off_diag = qml.math.cast_like(qml.math.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], like=js), 1j)
        if qml.math.ndim(phi) == 0:
            return qml.math.diag([1, c, c, 1]) + js * off_diag
        ones = qml.math.ones_like(c)
        diags = stack_last([ones, c, c, ones])[:, :, np.newaxis]
        return diags * np.eye(4) + qml.math.tensordot(js, off_diag, axes=0)

    @staticmethod
    def compute_eigvals(phi):
        """Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \\Sigma U^{\\dagger},

        where :math:`\\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.IsingXY.eigvals`


        Args:
            phi (tensor_like or float): phase angle

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.IsingXY.compute_eigvals(0.5)
        array([0.96891242+0.24740396j, 0.96891242-0.24740396j,       1.        +0.j        , 1.        +0.j        ])
        """
        if qml.math.get_interface(phi) == 'tensorflow':
            phi = qml.math.cast_like(phi, 1j)
        signs = np.array([1, -1, 0, 0])
        if qml.math.ndim(phi) == 0:
            return qml.math.exp(0.5j * phi * signs)
        return qml.math.exp(qml.math.tensordot(0.5j * phi, signs, axes=0))

    def adjoint(self):
        phi, = self.parameters
        return IsingXY(-phi, wires=self.wires)

    def pow(self, z):
        return [IsingXY(self.data[0] * z, wires=self.wires)]

    def simplify(self):
        phi = self.data[0] % (4 * np.pi)
        if _can_replace(phi, 0):
            return qml.Identity(wires=self.wires[0])
        return IsingXY(phi, wires=self.wires)