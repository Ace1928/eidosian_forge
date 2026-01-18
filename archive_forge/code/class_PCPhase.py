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
class PCPhase(Operation):
    """PCPhase(phi, dim, wires)
    A projector-controlled phase gate.

    This gate applies a complex phase :math:`e^{i\\phi}` to the first :math:`dim`
    basis vectors of the input state while applying a complex phase :math:`e^{-i \\phi}`
    to the remaining basis vectors. For example, consider the 2-qubit case where :math:`dim = 3`:

    .. math:: \\Pi(\\phi) = \\begin{bmatrix}
                e^{i\\phi} & 0 & 0 & 0 \\\\
                0 & e^{i\\phi} & 0 & 0 \\\\
                0 & 0 & e^{i\\phi} & 0 \\\\
                0 & 0 & 0 & e^{-i\\phi}
            \\end{bmatrix}.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)

    Args:
        phi (float): rotation angle :math:`\\phi`
        dim (int): the dimension of the subspace
        wires (Iterable[int, str], Wires): the wires the operation acts on
        id (str or None): String representing the operation (optional)

    **Example:**

    We can define a circuit using :class:`~.PCPhase` as follows:

    >>> dev = qml.device('default.qubit', wires=2)
    >>> @qml.qnode(dev)
    >>> def example_circuit():
    ...     qml.PCPhase(0.27, dim = 2, wires=range(2))
    ...     return qml.state()

    The resulting operation applies a complex phase :math:`e^{0.27i}` to the first :math:`dim = 2`
    basis vectors and :math:`e^{-0.27i}` to the remaining basis vectors.

    >>> print(np.round(qml.matrix(example_circuit)(),2))
    [[0.96+0.27j 0.  +0.j   0.  +0.j   0.  +0.j  ]
     [0.  +0.j   0.96+0.27j 0.  +0.j   0.  +0.j  ]
     [0.  +0.j   0.  +0.j   0.96-0.27j 0.  +0.j  ]
     [0.  +0.j   0.  +0.j   0.  +0.j   0.96-0.27j]]

    We can also choose a different :math:`dim` value to apply the phase shift to a different set of
    basis vectors as follows:

    >>> pc_op = qml.PCPhase(1.23, dim=3, wires=[1, 2])
    >>> print(np.round(qml.matrix(pc_op),2))
    [[0.33+0.94j 0.  +0.j   0.  +0.j   0.  +0.j  ]
     [0.  +0.j   0.33+0.94j 0.  +0.j   0.  +0.j  ]
     [0.  +0.j   0.  +0.j   0.33+0.94j 0.  +0.j  ]
     [0.  +0.j   0.  +0.j   0.  +0.j   0.33-0.94j]]
    """
    num_wires = AnyWires
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (0,)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    basis = 'Z'
    grad_method = 'A'
    parameter_frequencies = [(2,)]

    def generator(self):
        dim, shape = self.hyperparameters['dimension']
        mat = np.diag([1 if index < dim else -1 for index in range(shape)])
        return qml.Hermitian(mat, wires=self.wires)

    def _flatten(self):
        hyperparameter = (('dim', self.hyperparameters['dimension'][0]),)
        return (tuple(self.data), (self.wires, hyperparameter))

    def __init__(self, phi, dim, wires, id=None):
        wires = wires if isinstance(wires, Wires) else Wires(wires)
        if not (isinstance(dim, int) and dim <= 2 ** len(wires)):
            raise ValueError(f'The projected dimension {dim} must be an integer that is less than or equal to the max size of the matrix {2 ** len(wires)}. Try adding more wires.')
        super().__init__(phi, wires=wires, id=id)
        self.hyperparameters['dimension'] = (dim, 2 ** len(wires))

    @staticmethod
    def compute_matrix(*params, **hyperparams):
        """Get the matrix representation of Pi-controlled phase unitary."""
        phi = params[0]
        d, t = hyperparams['dimension']
        if qml.math.get_interface(phi) == 'tensorflow':
            p = qml.math.exp(1j * qml.math.cast_like(phi, 1j))
            minus_p = qml.math.exp(-1j * qml.math.cast_like(phi, 1j))
            zeros = qml.math.zeros_like(p)
            columns = []
            for i in range(t):
                columns.append([p if j == i else zeros for j in range(t)] if i < d else [minus_p if j == i else zeros for j in range(t)])
            r = qml.math.stack(columns, like='tensorflow', axis=-2)
            return r
        arg = 1j * phi
        prefactors = qml.math.array([1 if index < d else -1 for index in range(t)], like=phi)
        if qml.math.ndim(arg) == 0:
            return qml.math.diag(qml.math.exp(arg * prefactors))
        diags = qml.math.exp(qml.math.outer(arg, prefactors))
        return qml.math.stack([qml.math.diag(d) for d in diags])

    @staticmethod
    def compute_eigvals(*params, **hyperparams):
        """Get the eigvals for the Pi-controlled phase unitary."""
        phi = params[0]
        d, t = hyperparams['dimension']
        if qml.math.get_interface(phi) == 'tensorflow':
            phase = qml.math.exp(1j * qml.math.cast_like(phi, 1j))
            minus_phase = qml.math.exp(-1j * qml.math.cast_like(phi, 1j))
            return stack_last([phase if index < d else minus_phase for index in range(t)])
        arg = 1j * phi
        prefactors = qml.math.array([1 if index < d else -1 for index in range(t)], like=phi)
        if qml.math.ndim(phi) == 0:
            product = arg * prefactors
        else:
            product = qml.math.outer(arg, prefactors)
        return qml.math.exp(product)

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparams):
        """Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \\dots O_n.

        .. note::

            Operations making up the decomposition should be queued within the
            ``compute_decomposition`` method.

        .. seealso:: :meth:`~.Operator.decomposition`.

        Args:
            *params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            wires (Iterable[Any], Wires): wires that the operator acts on
            **hyperparams (dict): non-trainable hyper-parameters of the operator, as stored in the ``hyperparameters`` attribute

        Returns:
            list[Operator]: decomposition of the operator
        """
        phi = params[0]
        k, n = hyperparams['dimension']

        def _get_op_from_binary_rep(binary_rep, theta, wires):
            if len(binary_rep) == 1:
                op = PhaseShift(theta, wires[0]) if int(binary_rep) else PauliX(wires[0]) @ PhaseShift(theta, wires[0]) @ PauliX(wires[0])
            else:
                base_op = PhaseShift(theta, wires[-1]) if int(binary_rep[-1]) else PauliX(wires[-1]) @ PhaseShift(theta, wires[-1]) @ PauliX(wires[-1])
                op = qml.ctrl(base_op, control=wires[:-1], control_values=[int(i) for i in binary_rep[:-1]])
            return op
        n_log2 = int(np.log2(n))
        positive_binary_reps = [bin(_k)[2:].zfill(n_log2) for _k in range(k)]
        negative_binary_reps = [bin(_k)[2:].zfill(n_log2) for _k in range(k, n)]
        positive_ops = [_get_op_from_binary_rep(br, phi, wires=wires) for br in positive_binary_reps]
        negative_ops = [_get_op_from_binary_rep(br, -1 * phi, wires=wires) for br in negative_binary_reps]
        return positive_ops + negative_ops

    def adjoint(self):
        """Computes the adjoint of the operator."""
        phi = self.parameters[0]
        dim, _ = self.hyperparameters['dimension']
        return PCPhase(-1 * phi, dim=dim, wires=self.wires)

    def pow(self, z):
        """Computes the operator raised to z."""
        phi = self.parameters[0]
        dim, _ = self.hyperparameters['dimension']
        return [PCPhase(phi * z, dim=dim, wires=self.wires)]

    def simplify(self):
        """Simplifies the operator if possible."""
        phi = self.parameters[0] % (2 * np.pi)
        dim, _ = self.hyperparameters['dimension']
        if _can_replace(phi, 0):
            return qml.Identity(wires=self.wires[0])
        return PCPhase(phi, dim=dim, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        """The label of the operator when displayed in a circuit."""
        return super().label(decimals=decimals, base_label=base_label or '∏_ϕ', cache=cache)