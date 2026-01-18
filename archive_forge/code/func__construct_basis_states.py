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
def _construct_basis_states(state_free, control_labels, target_label):
    e1 = []
    e2 = []
    j = 0
    for i in range(len(state_free) + len(control_labels) + 1):
        if i in control_labels:
            e1.append(1)
            e2.append(1)
        elif i == target_label:
            e1.append(0)
            e2.append(1)
        else:
            e1.append(state_free[j])
            e2.append(state_free[j])
            j += 1
    out1 = _bin_to_int(e1)
    out2 = _bin_to_int(e2)
    return (out1, out2)