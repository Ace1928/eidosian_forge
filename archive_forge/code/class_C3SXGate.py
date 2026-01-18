from __future__ import annotations
from typing import Optional, Union, Type
from math import ceil, pi
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import _ctrl_state_to_int, with_gate_array, with_controlled_gate_array
class C3SXGate(SingletonControlledGate):
    """The 3-qubit controlled sqrt-X gate.

    This implementation is based on Page 17 of [1].

    References:
        [1] Barenco et al., 1995. https://arxiv.org/pdf/quant-ph/9503016.pdf
    """

    def __init__(self, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, *, duration=None, unit='dt', _base_label=None):
        """Create a new 3-qubit controlled sqrt-X gate.

        Args:
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g.``'110'``), or ``None``. If ``None``, use all 1s.
        """
        from .sx import SXGate
        super().__init__('c3sx', 4, [], num_ctrl_qubits=3, label=label, ctrl_state=ctrl_state, base_gate=SXGate(label=_base_label), duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=3)

    def _define(self):
        """
        gate c3sqrtx a,b,c,d
        {
            h d; cu1(pi/8) a,d; h d;
            cx a,b;
            h d; cu1(-pi/8) b,d; h d;
            cx a,b;
            h d; cu1(pi/8) b,d; h d;
            cx b,c;
            h d; cu1(-pi/8) c,d; h d;
            cx a,c;
            h d; cu1(pi/8) c,d; h d;
            cx b,c;
            h d; cu1(-pi/8) c,d; h d;
            cx a,c;
            h d; cu1(pi/8) c,d; h d;
        }
        """
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import CU1Gate
        from .h import HGate
        angle = numpy.pi / 8
        q = QuantumRegister(4, name='q')
        rules = [(HGate(), [q[3]], []), (CU1Gate(angle), [q[0], q[3]], []), (HGate(), [q[3]], []), (CXGate(), [q[0], q[1]], []), (HGate(), [q[3]], []), (CU1Gate(-angle), [q[1], q[3]], []), (HGate(), [q[3]], []), (CXGate(), [q[0], q[1]], []), (HGate(), [q[3]], []), (CU1Gate(angle), [q[1], q[3]], []), (HGate(), [q[3]], []), (CXGate(), [q[1], q[2]], []), (HGate(), [q[3]], []), (CU1Gate(-angle), [q[2], q[3]], []), (HGate(), [q[3]], []), (CXGate(), [q[0], q[2]], []), (HGate(), [q[3]], []), (CU1Gate(angle), [q[2], q[3]], []), (HGate(), [q[3]], []), (CXGate(), [q[1], q[2]], []), (HGate(), [q[3]], []), (CU1Gate(-angle), [q[2], q[3]], []), (HGate(), [q[3]], []), (CXGate(), [q[0], q[2]], []), (HGate(), [q[3]], []), (CU1Gate(angle), [q[2], q[3]], []), (HGate(), [q[3]], [])]
        qc = QuantumCircuit(q)
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def __eq__(self, other):
        return isinstance(other, C3SXGate) and self.ctrl_state == other.ctrl_state