from __future__ import annotations
import cmath
import math
import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates.h import HGate
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.exceptions import QiskitError
from qiskit.synthesis.one_qubit.one_qubit_decompose import OneQubitEulerDecomposer
from .diagonal import Diagonal
def _demultiplex_single_uc(self, a, b):
    """
        This method implements the decomposition given in equation (3) in
        https://arxiv.org/pdf/quant-ph/0410066.pdf.
        The decomposition is used recursively to decompose uniformly controlled gates.
        a,b = single qubit unitaries
        v,u,r = outcome of the decomposition given in the reference mentioned above
        (see there for the details).
        """
    x = a.dot(UCGate._ct(b))
    det_x = np.linalg.det(x)
    x11 = x.item((0, 0)) / cmath.sqrt(det_x)
    phi = cmath.phase(det_x)
    r1 = cmath.exp(1j / 2 * (np.pi / 2 - phi / 2 - cmath.phase(x11)))
    r2 = cmath.exp(1j / 2 * (np.pi / 2 - phi / 2 + cmath.phase(x11) + np.pi))
    r = np.array([[r1, 0], [0, r2]], dtype=complex)
    d, u = np.linalg.eig(r.dot(x).dot(r))
    if abs(d[0] + 1j) < _EPS:
        d = np.flip(d, 0)
        u = np.flip(u, 1)
    d = np.diag(np.sqrt(d))
    v = d.dot(UCGate._ct(u)).dot(UCGate._ct(r)).dot(b)
    return (v, u, r)