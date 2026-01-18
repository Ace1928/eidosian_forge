from __future__ import annotations
from typing import Optional, Union, Type
from math import ceil, pi
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import _ctrl_state_to_int, with_gate_array, with_controlled_gate_array
@with_controlled_gate_array(_X_ARRAY, num_ctrl_qubits=2, cached_states=(3,))
class CCXGate(SingletonControlledGate):
    """CCX gate, also known as Toffoli gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.ccx` and
    :meth:`~qiskit.circuit.QuantumCircuit.toffoli` methods.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ──■──
               │
        q_1: ──■──
             ┌─┴─┐
        q_2: ┤ X ├
             └───┘

    **Matrix representation:**

    .. math::

        CCX q_0, q_1, q_2 =
            I \\otimes I \\otimes |0 \\rangle \\langle 0| + CX \\otimes |1 \\rangle \\langle 1| =
           \\begin{pmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\\\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\\\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\\\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\\\\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\\\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\\\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\\\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0
            \\end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_2 and q_1. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌───┐
            q_0: ┤ X ├
                 └─┬─┘
            q_1: ──■──
                   │
            q_2: ──■──

        .. math::

            CCX\\ q_2, q_1, q_0 =
                |0 \\rangle \\langle 0| \\otimes I \\otimes I + |1 \\rangle \\langle 1| \\otimes CX =
                \\begin{pmatrix}
                    1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\\\
                    0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\\\
                    0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\\\
                    0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\\\
                    0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\\\
                    0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\\\
                    0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\\\\
                    0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
                \\end{pmatrix}

    """

    def __init__(self, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, *, duration=None, unit='dt', _base_label=None):
        """Create new CCX gate."""
        super().__init__('ccx', 3, [], num_ctrl_qubits=2, label=label, ctrl_state=ctrl_state, base_gate=XGate(label=_base_label), duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=2)

    def _define(self):
        """
        gate ccx a,b,c
        {
        h c; cx b,c; tdg c; cx a,c;
        t c; cx b,c; tdg c; cx a,c;
        t b; t c; h c; cx a,b;
        t a; tdg b; cx a,b;}
        """
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .h import HGate
        from .t import TGate, TdgGate
        q = QuantumRegister(3, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(HGate(), [q[2]], []), (CXGate(), [q[1], q[2]], []), (TdgGate(), [q[2]], []), (CXGate(), [q[0], q[2]], []), (TGate(), [q[2]], []), (CXGate(), [q[1], q[2]], []), (TdgGate(), [q[2]], []), (CXGate(), [q[0], q[2]], []), (TGate(), [q[1]], []), (TGate(), [q[2]], []), (HGate(), [q[2]], []), (CXGate(), [q[0], q[1]], []), (TGate(), [q[0]], []), (TdgGate(), [q[1]], []), (CXGate(), [q[0], q[1]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
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
            gate = MCXGate(num_ctrl_qubits=num_ctrl_qubits + 2, label=label, ctrl_state=new_ctrl_state, _base_label=self.label)
        else:
            gate = super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state, annotated=annotated)
        return gate

    def inverse(self, annotated: bool=False):
        """Return an inverted CCX gate (also a CCX).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            CCXGate: inverse gate (self-inverse).
        """
        return CCXGate(ctrl_state=self.ctrl_state)

    def __eq__(self, other):
        return isinstance(other, CCXGate) and self.ctrl_state == other.ctrl_state