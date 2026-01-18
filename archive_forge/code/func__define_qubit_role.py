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
def _define_qubit_role(self, q):
    n = int(np.log2(self.iso_data.shape[0]))
    m = int(np.log2(self.iso_data.shape[1]))
    q_input = q[:m]
    q_ancillas_for_output = q[m:n]
    q_ancillas_zero = q[n:n + self.num_ancillas_zero]
    q_ancillas_dirty = q[n + self.num_ancillas_zero:]
    return (q_input, q_ancillas_for_output, q_ancillas_zero, q_ancillas_dirty)