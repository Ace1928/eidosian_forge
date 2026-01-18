import pennylane as qml
from pennylane.operation import AnyWires, Operation
class HilbertSchmidt(Operation):
    """Create a Hilbert-Schmidt template that can be used to compute the Hilbert-Schmidt Test (HST).

    The HST is a useful quantity used when we want to compile an unitary `U` with an approximate unitary `V`. The HST
    is used as a distance between `U` and `V`, the result of executing the HST is 0 if and only if `V` is equal to
    `U` (up to a global phase). Therefore we can define a cost by:

    .. math::
        C_{HST} = 1 - \\frac{1}{d^2} \\left|Tr(V^{\\dagger}U)\\right|^2,

    where the quantity :math:`\\frac{1}{d^2} \\left|Tr(V^{\\dagger}U)\\right|^2` is obtained by executing the
    Hilbert-Schmidt Test. It is equivalent to taking the outcome probability of the state :math:`|0 ... 0\\rangle`
    for the following circuit:

    .. figure:: ../../_static/templates/subroutines/hst.png
        :align: center
        :width: 80%
        :target: javascript:void(0);

    It defines our decomposition for the Hilbert-Schmidt Test template.

    Args:
        *params (array): Parameters for the quantum function `V`.
        v_function (callable): Quantum function that represents the approximate compiled unitary `V`.
        v_wires (int or Iterable[Number, str]]): The wire(s) the approximate compiled unitary act on.
        u_tape (.QuantumTape): `U`, the unitary to be compiled as a ``qml.tape.QuantumTape``.

    Raises:
        QuantumFunctionError: The argument ``u_tape`` must be a ``QuantumTape``.
        QuantumFunctionError: ``v_function`` is not a valid quantum function.
        QuantumFunctionError: ``U`` and ``V`` do not have the same number of wires.
        QuantumFunctionError: The wires ``v_wires`` are a subset of ``V`` wires.
        QuantumFunctionError: ``u_tape`` and ``v_tape`` must act on distinct wires.

    **Reference**

    [1] Sumeet Khatri, Ryan LaRose, Alexander Poremba, Lukasz Cincio, Andrew T. Sornborger and Patrick J. Coles
    Quantum-assisted Quantum Compiling.
    `arxiv/1807.00800 <https://arxiv.org/pdf/1807.00800.pdf>`_

    .. seealso:: :class:`~.LocalHilbertSchmidt`

    .. details::
        :title: Usage Details

        Consider that we want to evaluate the Hilbert-Schmidt Test cost between the unitary ``U`` and an approximate
        unitary ``V``. We need to define some functions where it is possible to use the :class:`~.HilbertSchmidt`
        template. Here the considered unitary is ``Hadamard`` and we try to compute the cost for the approximate
        unitary ``RZ``. For an angle that is equal to ``0`` (``Identity``), we have the maximal cost which is ``1``.

        .. code-block:: python

            with qml.QueuingManager.stop_recording():
                u_tape = qml.tape.QuantumTape([qml.Hadamard(0)])

            def v_function(params):
                qml.RZ(params[0], wires=1)

            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev)
            def hilbert_test(v_params, v_function, v_wires, u_tape):
                qml.HilbertSchmidt(v_params, v_function=v_function, v_wires=v_wires, u_tape=u_tape)
                return qml.probs(u_tape.wires + v_wires)

            def cost_hst(parameters, v_function, v_wires, u_tape):
                return (1 - hilbert_test(v_params=parameters, v_function=v_function, v_wires=v_wires, u_tape=u_tape)[0])

        Now that the cost function has been defined it can be called for specific parameters:

        >>> cost_hst([0], v_function = v_function, v_wires = [1], u_tape = u_tape)
        1

    """
    num_wires = AnyWires
    grad_method = None

    def _flatten(self):
        metadata = (('v_function', self.hyperparameters['v_function']), ('v_wires', self.hyperparameters['v_wires']), ('u_tape', self.hyperparameters['u_tape']))
        return (self.data, metadata)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(*data, **dict(metadata))

    def __init__(self, *params, v_function, v_wires, u_tape, id=None):
        self._num_params = len(params)
        if not isinstance(u_tape, qml.tape.QuantumScript):
            raise qml.QuantumFunctionError('The argument u_tape must be a QuantumTape.')
        u_wires = u_tape.wires
        self.hyperparameters['u_tape'] = u_tape
        if not callable(v_function):
            raise qml.QuantumFunctionError('The argument v_function must be a callable quantum function.')
        self.hyperparameters['v_function'] = v_function
        v_tape = qml.tape.make_qscript(v_function)(*params)
        self.hyperparameters['v_tape'] = v_tape
        self.hyperparameters['v_wires'] = qml.wires.Wires(v_wires)
        if len(u_wires) != len(v_wires):
            raise qml.QuantumFunctionError('U and V must have the same number of wires.')
        if not qml.wires.Wires(v_wires).contains_wires(v_tape.wires):
            raise qml.QuantumFunctionError('All wires in v_tape must be in v_wires.')
        if len(qml.wires.Wires.shared_wires([u_tape.wires, v_tape.wires])) != 0:
            raise qml.QuantumFunctionError('u_tape and v_tape must act on distinct wires.')
        wires = qml.wires.Wires(u_wires + v_wires)
        super().__init__(*params, wires=wires, id=id)

    @property
    def num_params(self):
        return self._num_params

    @staticmethod
    def compute_decomposition(params, wires, u_tape, v_tape, v_function=None, v_wires=None):
        """Representation of the operator as a product of other operators."""
        n_wires = len(u_tape.wires + v_tape.wires)
        first_range = range(n_wires // 2)
        second_range = range(n_wires // 2, n_wires)
        decomp_ops = [qml.Hadamard(wires[i]) for i in first_range]
        decomp_ops.extend((qml.CNOT(wires=[wires[i], wires[j]]) for i, j in zip(first_range, second_range)))
        for op_u in u_tape.operations:
            if qml.QueuingManager.recording():
                qml.apply(op_u)
            decomp_ops.append(op_u)
        decomp_ops.extend((qml.adjoint(op_v, lazy=False) for op_v in v_tape.operations))
        decomp_ops.extend((qml.CNOT(wires=[wires[i], wires[j]]) for i, j in zip(reversed(first_range), reversed(second_range))))
        decomp_ops.extend((qml.Hadamard(wires[i]) for i in first_range))
        return decomp_ops