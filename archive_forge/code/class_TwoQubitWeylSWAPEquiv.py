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
class TwoQubitWeylSWAPEquiv(TwoQubitWeylDecomposition):
    """:math:`U \\sim U_d(\\pi/4, \\pi/4, \\pi/4) \\sim U(\\pi/4, \\pi/4, -\\pi/4) \\sim \\text{SWAP}`

    This gate binds 0 parameters, we make it canonical by setting
    :math:`K2_l = Id` , :math:`K2_r = Id`.
    """

    def specialize(self):
        if self.c > 0:
            self.K1l = self.K1l @ self.K2r
            self.K1r = self.K1r @ self.K2l
        else:
            self._is_flipped_from_original = True
            self.K1l = self.K1l @ _ipz @ self.K2r
            self.K1r = self.K1r @ _ipz @ self.K2l
            self.global_phase = self.global_phase + np.pi / 2
        self.a = self.b = self.c = np.pi / 4
        self.K2l = _id.copy()
        self.K2r = _id.copy()

    def _weyl_gate(self, simplify, circ: QuantumCircuit, atol):
        del self, simplify, atol
        circ.swap(0, 1)
        circ.global_phase -= 3 * np.pi / 4