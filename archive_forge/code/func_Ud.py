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
def Ud(a, b, c):
    """Generates the array :math:`e^{(i a XX + i b YY + i c ZZ)}`"""
    return np.array([[cmath.exp(1j * c) * math.cos(a - b), 0, 0, 1j * cmath.exp(1j * c) * math.sin(a - b)], [0, cmath.exp(-1j * c) * math.cos(a + b), 1j * cmath.exp(-1j * c) * math.sin(a + b), 0], [0, 1j * cmath.exp(-1j * c) * math.sin(a + b), cmath.exp(-1j * c) * math.cos(a + b), 0], [1j * cmath.exp(1j * c) * math.sin(a - b), 0, 0, cmath.exp(1j * c) * math.cos(a - b)]], dtype=complex)