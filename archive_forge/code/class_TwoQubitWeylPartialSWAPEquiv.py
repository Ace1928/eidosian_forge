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
class TwoQubitWeylPartialSWAPEquiv(TwoQubitWeylDecomposition):
    """:math:`U \\sim U_d(\\alpha\\pi/4, \\alpha\\pi/4, \\alpha\\pi/4) \\sim \\text{SWAP}^\\alpha`
    This gate binds 3 parameters, we make it canonical by setting:
    :math:`K2_l = Id`.
    """

    def specialize(self):
        self.a = self.b = self.c = _closest_partial_swap(self.a, self.b, self.c)
        self.K1l = self.K1l @ self.K2l
        self.K1r = self.K1r @ self.K2l
        self.K2r = self.K2l.T.conj() @ self.K2r
        self.K2l = _id.copy()