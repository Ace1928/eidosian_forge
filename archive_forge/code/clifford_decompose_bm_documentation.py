from itertools import product
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
Return CX cost of a 3-qubit clifford.