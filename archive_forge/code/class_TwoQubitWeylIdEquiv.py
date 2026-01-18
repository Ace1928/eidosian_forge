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
class TwoQubitWeylIdEquiv(TwoQubitWeylDecomposition):
    """:math:`U \\sim U_d(0,0,0) \\sim Id`

    This gate binds 0 parameters, we make it canonical by setting
    :math:`K2_l = Id` , :math:`K2_r = Id`.
    """

    def specialize(self):
        self.a = self.b = self.c = 0.0
        self.K1l = self.K1l @ self.K2l
        self.K1r = self.K1r @ self.K2r
        self.K2l = _id.copy()
        self.K2r = _id.copy()