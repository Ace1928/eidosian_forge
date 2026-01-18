from __future__ import annotations
from cmath import exp
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType
class MCPhaseGate(ControlledGate):
    """Multi-controlled-Phase gate.

    This is a diagonal and symmetric gate that induces a
    phase on the state of the target qubit, depending on the state of the control qubits.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.mcp` method.

    **Circuit symbol:**

    .. parsed-literal::

            q_0: ───■────
                    │
                    .
                    │
        q_(n-1): ───■────
                 ┌──┴───┐
            q_n: ┤ P(λ) ├
                 └──────┘

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.CPhaseGate`:
        The singly-controlled-version of this gate.
    """

    def __init__(self, lam: ParameterValueType, num_ctrl_qubits: int, label: str | None=None, *, duration=None, unit='dt', _base_label=None):
        """Create new MCPhase gate."""
        super().__init__('mcphase', num_ctrl_qubits + 1, [lam], num_ctrl_qubits=num_ctrl_qubits, label=label, base_gate=PhaseGate(lam, label=_base_label), duration=duration, unit=unit)

    def _define(self):
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        q = QuantumRegister(self.num_qubits, 'q')
        qc = QuantumCircuit(q, name=self.name)
        if self.num_ctrl_qubits == 0:
            qc.p(self.params[0], 0)
        if self.num_ctrl_qubits == 1:
            qc.cp(self.params[0], 0, 1)
        else:
            from .u3 import _gray_code_chain
            scaled_lam = self.params[0] / 2 ** (self.num_ctrl_qubits - 1)
            bottom_gate = CPhaseGate(scaled_lam)
            for operation, qubits, clbits in _gray_code_chain(q, self.num_ctrl_qubits, bottom_gate):
                qc._append(operation, qubits, clbits)
        self.definition = qc

    def control(self, num_ctrl_qubits: int=1, label: str | None=None, ctrl_state: str | int | None=None, annotated: bool=False):
        """Controlled version of this gate.

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
        if not annotated and ctrl_state is None:
            gate = MCPhaseGate(self.params[0], num_ctrl_qubits=num_ctrl_qubits + self.num_ctrl_qubits, label=label)
            gate.base_gate.label = self.label
        else:
            gate = super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state, annotated=annotated)
        return gate

    def inverse(self, annotated: bool=False):
        """Return inverted MCU1 gate (:math:`MCU1(\\lambda)^{\\dagger} = MCU1(-\\lambda)`)"""
        return MCPhaseGate(-self.params[0], self.num_ctrl_qubits)