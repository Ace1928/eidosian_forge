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
class TwoQubitDecomposeUpToDiagonal:
    """
    Class to decompose two qubit unitaries into the product of a diagonal gate
    and another unitary gate which can be represented by two CX gates instead of the
    usual three. This can be used when neighboring gates commute with the diagonal to
    potentially reduce overall CX count.
    """

    def __init__(self):
        sy = np.array([[0, -1j], [1j, 0]])
        self.sysy = np.kron(sy, sy)

    def _u4_to_su4(self, u4):
        phase_factor = np.conj(np.linalg.det(u4) ** (-1 / u4.shape[0]))
        su4 = u4 / phase_factor
        return (su4, cmath.phase(phase_factor))

    def _gamma(self, mat):
        """
        proposition II.1: this invariant characterizes when two operators in U(4),
        say u, v, are equivalent up to single qubit gates:

           u ≡ v -> Det(γ(u)) = Det(±(γ(v)))
        """
        sumat, _ = self._u4_to_su4(mat)
        sysy = self.sysy
        return sumat @ sysy @ sumat.T @ sysy

    def _cx0_test(self, mat):
        gamma = self._gamma(mat)
        evals = np.linalg.eigvals(gamma)
        return np.all(np.isclose(evals, np.ones(4)))

    def _cx1_test(self, mat):
        gamma = self._gamma(mat)
        evals = np.linalg.eigvals(gamma)
        uvals, ucnts = np.unique(np.round(evals, 10), return_counts=True)
        return len(uvals) == 2 and all(ucnts == 2) and all((np.isclose(x, 1j) or np.isclose(x, -1j) for x in uvals))

    def _cx2_test(self, mat):
        gamma = self._gamma(mat)
        return np.isclose(np.trace(gamma).imag, 0)

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

    def __call__(self, mat):
        """do the decomposition"""
        su4, phase = self._u4_to_su4(mat)
        real_map = self._real_trace_transform(su4)
        mapped_su4 = real_map @ su4
        if not self._cx2_test(mapped_su4):
            warnings.warn('Unitary decomposition up to diagonal may use an additionl CX gate.')
        circ = two_qubit_cnot_decompose(mapped_su4)
        circ.global_phase += phase
        return (real_map.conj(), circ)