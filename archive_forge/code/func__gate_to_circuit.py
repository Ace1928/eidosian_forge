from __future__ import annotations
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import UnitaryGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.basis import BasisTranslator, UnrollCustomDefinitions
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from . import ControlledGate, Gate, QuantumRegister, QuantumCircuit
from ._utils import _ctrl_state_to_int
def _gate_to_circuit(operation):
    """Converts a gate instance to a QuantumCircuit"""
    if hasattr(operation, 'definition') and operation.definition is not None:
        return operation.definition
    else:
        qr = QuantumRegister(operation.num_qubits)
        qc = QuantumCircuit(qr, name=operation.name)
        qc.append(operation, qr)
        return qc