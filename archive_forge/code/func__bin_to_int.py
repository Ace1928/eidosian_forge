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
def _bin_to_int(binary_digits_as_list):
    return int(''.join((str(x) for x in binary_digits_as_list)), 2)