from __future__ import annotations
from collections.abc import Callable
from qiskit import circuit
from qiskit.circuit import ControlledGate, Gate, QuantumRegister, QuantumCircuit
from qiskit.exceptions import QiskitError
from ..standard_gates import XGate, YGate, ZGate, HGate, TGate, TdgGate, SGate, SdgGate
def _identify_gate(self, gate):
    """Case the gate input to a gate."""
    valid_gates = {'ch': HGate(), 'cx': XGate(), 'cy': YGate(), 'cz': ZGate(), 'h': HGate(), 's': SGate(), 'sdg': SdgGate(), 'x': XGate(), 'y': YGate(), 'z': ZGate(), 't': TGate(), 'tdg': TdgGate()}
    if isinstance(gate, ControlledGate):
        base_gate = gate.base_gate
    elif isinstance(gate, Gate):
        if gate.num_qubits != 1:
            raise AttributeError('Base gate must act on one qubit only.')
        base_gate = gate
    elif isinstance(gate, QuantumCircuit):
        if gate.num_qubits != 1:
            raise AttributeError('The circuit you specified as control gate can only have one qubit!')
        base_gate = gate.to_gate()
    else:
        if callable(gate):
            name = gate.__name__
        elif isinstance(gate, str):
            name = gate
        else:
            raise AttributeError(f'Invalid gate specified: {gate}')
        base_gate = valid_gates[name]
    return base_gate