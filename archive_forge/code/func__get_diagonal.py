import numpy as np
from qiskit.circuit import Gate
from qiskit.circuit.quantumcircuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_isometry
from .uc import UCGate
def _get_diagonal(self):
    _, diag = self._dec_mcg_up_diag()
    return diag