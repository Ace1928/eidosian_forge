from itertools import product
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
def _reduce_cost(clifford, inv_circuit, cost):
    """Two-qubit cost reduction step"""
    num_qubits = clifford.num_qubits
    for qubit0 in range(num_qubits):
        for qubit1 in range(qubit0 + 1, num_qubits):
            for n0, n1 in product(range(3), repeat=2):
                reduced = clifford.copy()
                for qubit, n in [(qubit0, n0), (qubit1, n1)]:
                    if n == 1:
                        _append_v(reduced, qubit)
                    elif n == 2:
                        _append_w(reduced, qubit)
                _append_cx(reduced, qubit0, qubit1)
                new_cost = _cx_cost(reduced)
                if new_cost == cost - 1:
                    for qubit, n in [(qubit0, n0), (qubit1, n1)]:
                        if n == 1:
                            inv_circuit.sdg(qubit)
                            inv_circuit.h(qubit)
                        elif n == 2:
                            inv_circuit.h(qubit)
                            inv_circuit.s(qubit)
                    inv_circuit.cx(qubit0, qubit1)
                    return (reduced, inv_circuit, new_cost)
    raise QiskitError('Failed to reduce Clifford CX cost.')