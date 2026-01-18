from __future__ import annotations
import itertools
import numpy as np
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_isometry
from .diagonal import Diagonal
from .uc import UCGate
from .mcg_up_to_diagonal import MCGupDiag
def _find_squs_for_disentangling(self, v, k, s):
    k_prime = 0
    n = int(np.log2(self.iso_data.shape[0]))
    if _b(k, s + 1) == 0:
        i_start = _a(k, s + 1)
    else:
        i_start = _a(k, s + 1) + 1
    id_list = [np.eye(2, 2) for _ in range(i_start)]
    squs = [_reverse_qubit_state([v[2 * i * 2 ** s + _b(k, s), k_prime], v[(2 * i + 1) * 2 ** s + _b(k, s), k_prime]], _k_s(k, s), self._epsilon) for i in range(i_start, 2 ** (n - s - 1))]
    return id_list + squs