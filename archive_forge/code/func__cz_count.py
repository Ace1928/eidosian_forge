from typing import Callable
from cirq import ops, circuits, transformers
from cirq.contrib.paulistring.pauli_string_optimize import pauli_string_optimized_circuit
from cirq.contrib.paulistring.clifford_optimize import clifford_optimized_circuit
def _cz_count(circuit):
    return sum((isinstance(op.gate, ops.CZPowGate) for moment in circuit for op in moment))