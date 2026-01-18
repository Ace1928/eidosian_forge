from __future__ import annotations
import cmath
import math
import io
import base64
import warnings
from typing import ClassVar, Optional, Type
import logging
import numpy as np
from qiskit.circuit import QuantumRegister, QuantumCircuit, Gate
from qiskit.circuit.library.standard_gates import CXGate, RXGate, RYGate, RZGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit.synthesis.two_qubit.weyl import transform_to_magic_basis
from qiskit.synthesis.one_qubit.one_qubit_decompose import (
from qiskit._accelerate import two_qubit_decompose
def _real_trace_transform(self, mat):
    """
        Determine diagonal gate such that

        U3 = D U2

        Where U3 is a general two-qubit gate which takes 3 cnots, D is a
        diagonal gate, and U2 is a gate which takes 2 cnots.
        """
    a1 = -mat[1, 3] * mat[2, 0] + mat[1, 2] * mat[2, 1] + mat[1, 1] * mat[2, 2] - mat[1, 0] * mat[2, 3]
    a2 = mat[0, 3] * mat[3, 0] - mat[0, 2] * mat[3, 1] - mat[0, 1] * mat[3, 2] + mat[0, 0] * mat[3, 3]
    theta = 0
    phi = 0
    psi = np.arctan2(a1.imag + a2.imag, a1.real - a2.real) - phi
    diag = np.diag(np.exp(-1j * np.array([theta, phi, psi, -(theta + phi + psi)])))
    return diag