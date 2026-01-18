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
def _decompose_column(self, circuit, q, diag, remaining_isometry, column_index):
    """
        Decomposes the column with index column_index.
        """
    n = int(np.log2(self.iso_data.shape[0]))
    for s in range(n):
        self._disentangle(circuit, q, diag, remaining_isometry, column_index, s)