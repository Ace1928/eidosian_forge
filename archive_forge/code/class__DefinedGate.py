import dataclasses
import math
from typing import Iterable, Callable
from qiskit.circuit import (
from qiskit._qasm2 import (  # pylint: disable=no-name-in-module
from .exceptions import QASM2ParseError
class _DefinedGate(Gate):
    """A gate object defined by a `gate` statement in an OpenQASM 2 program.  This object lazily
    binds its parameters to its definition, so it is only synthesised when required."""

    def __init__(self, name, num_qubits, params, gates, bytecode):
        self._gates = gates
        self._bytecode = bytecode
        super().__init__(name, num_qubits, list(params))

    def _define(self):
        qubits = [Qubit() for _ in [None] * self.num_qubits]
        qc = QuantumCircuit(qubits)
        for op in self._bytecode:
            if op.opcode == OpCode.Gate:
                gate_id, args, op_qubits = op.operands
                qc._append(CircuitInstruction(self._gates[gate_id](*(_evaluate_argument(a, self.params) for a in args)), [qubits[q] for q in op_qubits]))
            elif op.opcode == OpCode.Barrier:
                op_qubits = op.operands[0]
                qc._append(CircuitInstruction(Barrier(len(op_qubits)), [qubits[q] for q in op_qubits]))
            else:
                raise ValueError(f'received invalid bytecode to build gate: {op}')
        self._definition = qc

    def __getstate__(self):
        return (self.name, self.num_qubits, self.params, self.definition, self.condition)

    def __setstate__(self, state):
        name, num_qubits, params, definition, condition = state
        super().__init__(name, num_qubits, params)
        self._gates = ()
        self._bytecode = ()
        self._definition = definition
        self._condition = condition