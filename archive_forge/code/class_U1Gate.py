from __future__ import annotations
from cmath import exp
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import _ctrl_state_to_int
class U1Gate(Gate):
    """Single-qubit rotation about the Z axis.

    This is a diagonal gate. It can be implemented virtually in hardware
    via framechanges (i.e. at zero error and duration).

    .. warning::

       This gate is deprecated. Instead, the following replacements should be used

       .. math::

           U1(\\lambda) = P(\\lambda)= U(0,0,\\lambda)

       .. code-block:: python

          circuit = QuantumCircuit(1)
          circuit.p(lambda, 0) # or circuit.u(0, 0, lambda)




    **Circuit symbol:**

    .. parsed-literal::

             ┌───────┐
        q_0: ┤ U1(λ) ├
             └───────┘

    **Matrix Representation:**

    .. math::

        U1(\\lambda) =
            \\begin{pmatrix}
                1 & 0 \\\\
                0 & e^{i\\lambda}
            \\end{pmatrix}

    **Examples:**

        .. math::

            U1(\\lambda = \\pi) = Z

        .. math::

            U1(\\lambda = \\pi/2) = S

        .. math::

            U1(\\lambda = \\pi/4) = T

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.RZGate`:
        This gate is equivalent to RZ up to a phase factor.

            .. math::

                U1(\\lambda) = e^{i{\\lambda}/2} RZ(\\lambda)

        :class:`~qiskit.circuit.library.standard_gates.U3Gate`:
        U3 is a generalization of U2 that covers all single-qubit rotations,
        using two X90 pulses.

        Reference for virtual Z gate implementation:
        `1612.00858 <https://arxiv.org/abs/1612.00858>`_
    """

    def __init__(self, theta: ParameterValueType, label: str | None=None, *, duration=None, unit='dt'):
        """Create new U1 gate."""
        super().__init__('u1', 1, [theta], label=label, duration=duration, unit=unit)

    def _define(self):
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u3 import U3Gate
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U3Gate(0, 0, self.params[0]), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def control(self, num_ctrl_qubits: int=1, label: str | None=None, ctrl_state: str | int | None=None, annotated: bool=False):
        """Return a (multi-)controlled-U1 gate.

        Args:
            num_ctrl_qubits: number of control qubits.
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g.``'110'``), or ``None``. If ``None``, use all 1s.
            annotated: indicates whether the controlled gate can be implemented
                as an annotated gate.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if not annotated and num_ctrl_qubits == 1:
            gate = CU1Gate(self.params[0], label=label, ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
        elif not annotated and ctrl_state is None and (num_ctrl_qubits > 1):
            gate = MCU1Gate(self.params[0], num_ctrl_qubits, label=label)
            gate.base_gate.label = self.label
        else:
            gate = super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state, annotated=annotated)
        return gate

    def inverse(self, annotated: bool=False):
        """Return inverted U1 gate (:math:`U1(\\lambda)^{\\dagger} = U1(-\\lambda))`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.U1Gate` with inverse parameter values.

        Returns:
            U1Gate: inverse gate.
        """
        return U1Gate(-self.params[0])

    def __array__(self, dtype=None):
        """Return a numpy.array for the U1 gate."""
        lam = float(self.params[0])
        return numpy.array([[1, 0], [0, numpy.exp(1j * lam)]], dtype=dtype)