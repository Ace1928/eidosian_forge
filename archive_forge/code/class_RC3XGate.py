from __future__ import annotations
from typing import Optional, Union, Type
from math import ceil, pi
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import _ctrl_state_to_int, with_gate_array, with_controlled_gate_array
@with_gate_array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1j, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0]])
class RC3XGate(SingletonGate):
    """The simplified 3-controlled Toffoli gate.

    The simplified Toffoli gate implements the Toffoli gate up to relative phases.
    Note, that the simplified Toffoli is not equivalent to the Toffoli. But can be used in places
    where the Toffoli gate is uncomputed again.

    This concrete implementation is from https://arxiv.org/abs/1508.03273, the complete circuit
    of Fig. 4.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.rcccx` method.
    """

    def __init__(self, label: Optional[str]=None, *, duration=None, unit='dt'):
        """Create a new RC3X gate."""
        super().__init__('rcccx', 4, [], label=label, duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """
        gate rc3x a,b,c,d
        { u2(0,pi) d;
          u1(pi/4) d;
          cx c,d;
          u1(-pi/4) d;
          u2(0,pi) d;
          cx a,d;
          u1(pi/4) d;
          cx b,d;
          u1(-pi/4) d;
          cx a,d;
          u1(pi/4) d;
          cx b,d;
          u1(-pi/4) d;
          u2(0,pi) d;
          u1(pi/4) d;
          cx c,d;
          u1(-pi/4) d;
          u2(0,pi) d;
        }
        """
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import U1Gate
        from .u2 import U2Gate
        q = QuantumRegister(4, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U2Gate(0, pi), [q[3]], []), (U1Gate(pi / 4), [q[3]], []), (CXGate(), [q[2], q[3]], []), (U1Gate(-pi / 4), [q[3]], []), (U2Gate(0, pi), [q[3]], []), (CXGate(), [q[0], q[3]], []), (U1Gate(pi / 4), [q[3]], []), (CXGate(), [q[1], q[3]], []), (U1Gate(-pi / 4), [q[3]], []), (CXGate(), [q[0], q[3]], []), (U1Gate(pi / 4), [q[3]], []), (CXGate(), [q[1], q[3]], []), (U1Gate(-pi / 4), [q[3]], []), (U2Gate(0, pi), [q[3]], []), (U1Gate(pi / 4), [q[3]], []), (CXGate(), [q[2], q[3]], []), (U1Gate(-pi / 4), [q[3]], []), (U2Gate(0, pi), [q[3]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def __eq__(self, other):
        return isinstance(other, RC3XGate)