import copy
import numpy as np
import pennylane as qml
from pennylane.queuing import QueuingManager
from pennylane.ops import BlockEncode, PCPhase
from pennylane.ops.op_math import adjoint
from pennylane.operation import AnyWires, Operation
class QSVT(Operation):
    """QSVT(UA,projectors)
    Implements the
    `quantum singular value transformation <https://arxiv.org/abs/1806.01838>`__ (QSVT) circuit.

    .. note ::

        This template allows users to define hardware-compatible block encoding and
        projector-controlled phase shift circuits. For a QSVT implementation that is
        tailored for simulators see :func:`~.qsvt` .

    Given an :class:`~.Operator` :math:`U`, which block encodes the matrix :math:`A`, and a list of
    projector-controlled phase shift operations :math:`\\vec{\\Pi}_\\phi`, this template applies a
    circuit for the quantum singular value transformation as follows.

    When the number of projector-controlled phase shifts is even (:math:`d` is odd), the QSVT
    circuit is defined as:

    .. math::

        U_{QSVT} = \\tilde{\\Pi}_{\\phi_1}U\\left[\\prod^{(d-1)/2}_{k=1}\\Pi_{\\phi_{2k}}U^\\dagger
        \\tilde{\\Pi}_{\\phi_{2k+1}}U\\right]\\Pi_{\\phi_{d+1}}.


    And when the number of projector-controlled phase shifts is odd (:math:`d` is even):

    .. math::

        U_{QSVT} = \\left[\\prod^{d/2}_{k=1}\\Pi_{\\phi_{2k-1}}U^\\dagger\\tilde{\\Pi}_{\\phi_{2k}}U\\right]
        \\Pi_{\\phi_{d+1}}.

    This circuit applies a polynomial transformation (:math:`Poly^{SV}`) to the singular values of
    the block encoded matrix:

    .. math::

        \\begin{align}
             U_{QSVT}(A, \\vec{\\phi}) &=
             \\begin{bmatrix}
                Poly^{SV}(A) & \\cdot \\\\
                \\cdot & \\cdot
            \\end{bmatrix}.
        \\end{align}

    .. seealso::

        :func:`~.qsvt` and `A Grand Unification of Quantum Algorithms <https://arxiv.org/pdf/2105.02859.pdf>`_.

    Args:
        UA (Operator): the block encoding circuit, specified as an :class:`~.Operator`,
            like :class:`~.BlockEncode`
        projectors (Sequence[Operator]): a list of projector-controlled phase
            shifts that implement the desired polynomial

    Raises:
        ValueError: if the input block encoding is not an operator

    **Example**

    To implement QSVT in a circuit, we can use the following method:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> A = np.array([[0.1]])
    >>> block_encode = qml.BlockEncode(A, wires=[0, 1])
    >>> shifts = [qml.PCPhase(i + 0.1, dim=1, wires=[0, 1]) for i in range(3)]
    >>> @qml.qnode(dev)
    >>> def example_circuit():
    ...    qml.QSVT(block_encode, shifts)
    ...    return qml.expval(qml.Z(0))

    The resulting circuit implements QSVT.

    >>> print(qml.draw(example_circuit)())
    0: ─╭QSVT─┤  <Z>
    1: ─╰QSVT─┤

    To see the implementation details, we can expand the circuit:

    >>> q_script = qml.tape.QuantumScript(ops=[qml.QSVT(block_encode, shifts)])
    >>> print(q_script.expand().draw(decimals=2))
    0: ─╭∏_ϕ(0.10)─╭BlockEncode(M0)─╭∏_ϕ(1.10)─╭BlockEncode(M0)†─╭∏_ϕ(2.10)─┤
    1: ─╰∏_ϕ(0.10)─╰BlockEncode(M0)─╰∏_ϕ(1.10)─╰BlockEncode(M0)†─╰∏_ϕ(2.10)─┤

    When working with this class directly, we can make use of any PennyLane operation
    to represent our block-encoding and our phase-shifts.

    >>> dev = qml.device("default.qubit", wires=[0])
    >>> block_encoding = qml.Hadamard(wires=0)  # note H is a block encoding of 1/sqrt(2)
    >>> phase_shifts = [qml.RZ(-2 * theta, wires=0) for theta in (1.23, -0.5, 4)]  # -2*theta to match convention
    >>>
    >>> @qml.qnode(dev)
    >>> def example_circuit():
    ...     qml.QSVT(block_encoding, phase_shifts)
    ...     return qml.expval(qml.Z(0))
    >>>
    >>> example_circuit()
    tensor(0.54030231, requires_grad=True)

    Once again, we can visualize the circuit as follows:

    >>> print(qml.draw(example_circuit)())
    0: ──QSVT─┤  <Z>

    To see the implementation details, we can expand the circuit:

    >>> q_script = qml.tape.QuantumScript(ops=[qml.QSVT(block_encoding, phase_shifts)])
    >>> print(q_script.expand().draw(decimals=2))
    0: ──RZ(-2.46)──H──RZ(1.00)──H†──RZ(-8.00)─┤
    """
    num_wires = AnyWires
    'int: Number of wires that the operator acts on.'
    grad_method = None
    'Gradient computation method.'

    def _flatten(self):
        data = (self.hyperparameters['UA'], self.hyperparameters['projectors'])
        return (data, tuple())

    @classmethod
    def _unflatten(cls, data, _) -> 'QSVT':
        return cls(*data)

    def __init__(self, UA, projectors, id=None):
        if not isinstance(UA, qml.operation.Operator):
            raise ValueError('Input block encoding must be an Operator')
        self._hyperparameters = {'UA': UA, 'projectors': projectors}
        ua_wires = UA.wires.toset()
        proj_wires = set.union(*(proj.wires.toset() for proj in projectors))
        total_wires = ua_wires.union(proj_wires)
        super().__init__(wires=total_wires, id=id)

    @property
    def data(self):
        """Flattened list of operator data in this QSVT operation.

        This ensures that the backend of a ``QuantumScript`` which contains a
        ``QSVT`` operation can be inferred with respect to the types of the
        ``QSVT`` block encoding and projector-controlled phase shift data.
        """
        return tuple((datum for op in self._operators for datum in op.data))

    @data.setter
    def data(self, new_data):
        if new_data:
            for op in self._operators:
                if op.num_params > 0:
                    op.data = new_data[:op.num_params]
                    new_data = new_data[op.num_params:]

    def __copy__(self):
        clone = QSVT.__new__(QSVT)
        clone._hyperparameters = {'UA': copy.copy(self._hyperparameters['UA']), 'projectors': list(map(copy.copy, self._hyperparameters['projectors']))}
        for attr, value in vars(self).items():
            if attr != '_hyperparameters':
                setattr(clone, attr, value)
        return clone

    @property
    def _operators(self) -> list[qml.operation.Operator]:
        """Flattened list of operators that compose this QSVT operation."""
        return [self._hyperparameters['UA'], *self._hyperparameters['projectors']]

    @staticmethod
    def compute_decomposition(*_data, UA, projectors, **_kwargs):
        """Representation of the operator as a product of other operators.

        The :class:`~.QSVT` is decomposed into alternating block encoding
        and projector-controlled phase shift operators. This is defined by the following
        equations, where :math:`U` is the block encoding operation and both :math:`\\Pi_\\phi` and
        :math:`\\tilde{\\Pi}_\\phi` are projector-controlled phase shifts with angle :math:`\\phi`.

        When the number of projector-controlled phase shifts is even (:math:`d` is odd), the QSVT
        circuit is defined as:

        .. math::

            U_{QSVT} = \\Pi_{\\phi_1}U\\left[\\prod^{(d-1)/2}_{k=1}\\Pi_{\\phi_{2k}}U^\\dagger
            \\tilde{\\Pi}_{\\phi_{2k+1}}U\\right]\\Pi_{\\phi_{d+1}}.


        And when the number of projector-controlled phase shifts is odd (:math:`d` is even):

        .. math::

            U_{QSVT} = \\left[\\prod^{d/2}_{k=1}\\Pi_{\\phi_{2k-1}}U^\\dagger\\tilde{\\Pi}_{\\phi_{2k}}U\\right]
            \\Pi_{\\phi_{d+1}}.

        .. seealso:: :meth:`~.QSVT.decomposition`.

        Args:
            UA (Operator): the block encoding circuit, specified as a :class:`~.Operator`
            projectors (list[Operator]): a list of projector-controlled phase
                shift circuits that implement the desired polynomial

        Returns:
            list[.Operator]: decomposition of the operator
        """
        op_list = []
        UA_adj = copy.copy(UA)
        for idx, op in enumerate(projectors[:-1]):
            if qml.QueuingManager.recording():
                qml.apply(op)
            op_list.append(op)
            if idx % 2 == 0:
                if qml.QueuingManager.recording():
                    qml.apply(UA)
                op_list.append(UA)
            else:
                op_list.append(adjoint(UA_adj))
        if qml.QueuingManager.recording():
            qml.apply(projectors[-1])
        op_list.append(projectors[-1])
        return op_list

    def label(self, decimals=None, base_label=None, cache=None):
        op_label = base_label or self.__class__.__name__
        return op_label

    def queue(self, context=QueuingManager):
        context.remove(self._hyperparameters['UA'])
        for op in self._hyperparameters['projectors']:
            context.remove(op)
        context.append(self)
        return self

    @staticmethod
    def compute_matrix(*args, **kwargs):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Operator.matrix` and :func:`~.matrix`

        Args:
            *params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            **hyperparams (dict): non-trainable hyperparameters of the operator, as stored in the ``hyperparameters`` attribute

        Returns:
            tensor_like: matrix representation
        """
        op_list = []
        UA = kwargs['UA']
        projectors = kwargs['projectors']
        with QueuingManager.stop_recording():
            UA_copy = copy.copy(UA)
            for idx, op in enumerate(projectors[:-1]):
                op_list.append(op)
                if idx % 2 == 0:
                    op_list.append(UA)
                else:
                    op_list.append(adjoint(UA_copy))
            op_list.append(projectors[-1])
            mat = qml.matrix(qml.prod(*tuple(op_list[::-1])))
        return mat