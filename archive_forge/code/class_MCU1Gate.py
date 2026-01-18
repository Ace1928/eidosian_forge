from __future__ import annotations
from cmath import exp
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import _ctrl_state_to_int
class MCU1Gate(ControlledGate):
    """Multi-controlled-U1 gate.

    This is a diagonal and symmetric gate that induces a
    phase on the state of the target qubit, depending on the state of the control qubits.

    **Circuit symbol:**

    .. parsed-literal::

            q_0: ────■────
                     │
                     .
                     │
        q_(n-1): ────■────
                 ┌───┴───┐
            q_n: ┤ U1(λ) ├
                 └───────┘

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.CU1Gate`:
        The singly-controlled-version of this gate.
    """

    def __init__(self, lam: ParameterValueType, num_ctrl_qubits: int, label: str | None=None, ctrl_state: str | int | None=None, *, duration=None, unit='dt', _base_label=None):
        """Create new MCU1 gate."""
        super().__init__('mcu1', num_ctrl_qubits + 1, [lam], num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state, base_gate=U1Gate(lam, label=_base_label), duration=duration, unit=unit)

    def _define(self):
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        q = QuantumRegister(self.num_qubits, 'q')
        qc = QuantumCircuit(q, name=self.name)
        if self.num_ctrl_qubits == 0:
            definition = U1Gate(self.params[0]).definition
        if self.num_ctrl_qubits == 1:
            definition = CU1Gate(self.params[0]).definition
        else:
            from .u3 import _gray_code_chain
            scaled_lam = self.params[0] / 2 ** (self.num_ctrl_qubits - 1)
            bottom_gate = CU1Gate(scaled_lam)
            definition = _gray_code_chain(q, self.num_ctrl_qubits, bottom_gate)
        for instr, qargs, cargs in definition:
            qc._append(instr, qargs, cargs)
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
        if not annotated:
            ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
            new_ctrl_state = self.ctrl_state << num_ctrl_qubits | ctrl_state
            gate = MCU1Gate(self.params[0], num_ctrl_qubits=num_ctrl_qubits + self.num_ctrl_qubits, label=label, ctrl_state=new_ctrl_state)
            gate.base_gate.label = self.label
        else:
            gate = super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state, annotated=annotated)
        return gate

    def inverse(self, annotated: bool=False):
        """Return inverted MCU1 gate (:math:`MCU1(\\lambda)^{\\dagger} = MCU1(-\\lambda))`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.MCU1Gate` with inverse
                parameter values.

        Returns:
            MCU1Gate: inverse gate.
        """
        return MCU1Gate(-self.params[0], self.num_ctrl_qubits)