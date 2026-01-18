from __future__ import annotations
from typing import Optional, Union, Type
from math import ceil, pi
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import _ctrl_state_to_int, with_gate_array, with_controlled_gate_array
@with_controlled_gate_array(_X_ARRAY, num_ctrl_qubits=3, cached_states=(7,))
class C3XGate(SingletonControlledGate):
    """The X gate controlled on 3 qubits.

    This implementation uses :math:`\\sqrt{T}` and 14 CNOT gates.
    """

    def __init__(self, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, *, _base_label=None, duration=None, unit='dt'):
        """Create a new 3-qubit controlled X gate."""
        super().__init__('mcx', 4, [], num_ctrl_qubits=3, label=label, ctrl_state=ctrl_state, base_gate=XGate(label=_base_label), duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=3)

    def _define(self):
        """
        gate c3x a,b,c,d
        {
            h d;
            p(pi/8) a;
            p(pi/8) b;
            p(pi/8) c;
            p(pi/8) d;
            cx a, b;
            p(-pi/8) b;
            cx a, b;
            cx b, c;
            p(-pi/8) c;
            cx a, c;
            p(pi/8) c;
            cx b, c;
            p(-pi/8) c;
            cx a, c;
            cx c, d;
            p(-pi/8) d;
            cx b, d;
            p(pi/8) d;
            cx c, d;
            p(-pi/8) d;
            cx a, d;
            p(pi/8) d;
            cx c, d;
            p(-pi/8) d;
            cx b, d;
            p(pi/8) d;
            cx c, d;
            p(-pi/8) d;
            cx a, d;
            h d;
        }
        """
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        q = QuantumRegister(4, name='q')
        qc = QuantumCircuit(q, name=self.name)
        qc.h(3)
        qc.p(pi / 8, [0, 1, 2, 3])
        qc.cx(0, 1)
        qc.p(-pi / 8, 1)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.p(-pi / 8, 2)
        qc.cx(0, 2)
        qc.p(pi / 8, 2)
        qc.cx(1, 2)
        qc.p(-pi / 8, 2)
        qc.cx(0, 2)
        qc.cx(2, 3)
        qc.p(-pi / 8, 3)
        qc.cx(1, 3)
        qc.p(pi / 8, 3)
        qc.cx(2, 3)
        qc.p(-pi / 8, 3)
        qc.cx(0, 3)
        qc.p(pi / 8, 3)
        qc.cx(2, 3)
        qc.p(-pi / 8, 3)
        qc.cx(1, 3)
        qc.p(pi / 8, 3)
        qc.cx(2, 3)
        qc.p(-pi / 8, 3)
        qc.cx(0, 3)
        qc.h(3)
        self.definition = qc

    def control(self, num_ctrl_qubits: int=1, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, annotated: bool=False):
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
            gate = MCXGate(num_ctrl_qubits=num_ctrl_qubits + 3, label=label, ctrl_state=new_ctrl_state, _base_label=self.label)
        else:
            gate = super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state, annotated=annotated)
        return gate

    def inverse(self, annotated: bool=False):
        """Invert this gate. The C3X is its own inverse.

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            C3XGate: inverse gate (self-inverse).
        """
        return C3XGate(ctrl_state=self.ctrl_state)

    def __eq__(self, other):
        return isinstance(other, C3XGate) and self.ctrl_state == other.ctrl_state