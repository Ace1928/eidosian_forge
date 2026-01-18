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
def _to_rxx_gate(self, angle: float) -> QuantumCircuit:
    """
        Takes an angle and returns the circuit equivalent to an RXXGate with the
        RXX equivalent gate as the two-qubit unitary.

        Args:
            angle: Rotation angle (in this case one of the Weyl parameters a, b, or c)

        Returns:
            Circuit: Circuit equivalent to an RXXGate.

        Raises:
            QiskitError: If the circuit is not equivalent to an RXXGate.
        """
    circ = QuantumCircuit(2)
    circ.append(self.rxx_equivalent_gate(self.scale * angle), qargs=[0, 1])
    decomposer_inv = TwoQubitWeylControlledEquiv(Operator(circ).data)
    oneq_decompose = OneQubitEulerDecomposer('ZYZ')
    rxx_circ = QuantumCircuit(2, global_phase=-decomposer_inv.global_phase)
    rxx_circ.compose(oneq_decompose(decomposer_inv.K2r).inverse(), inplace=True, qubits=[0])
    rxx_circ.compose(oneq_decompose(decomposer_inv.K2l).inverse(), inplace=True, qubits=[1])
    rxx_circ.compose(circ, inplace=True)
    rxx_circ.compose(oneq_decompose(decomposer_inv.K1r).inverse(), inplace=True, qubits=[0])
    rxx_circ.compose(oneq_decompose(decomposer_inv.K1l).inverse(), inplace=True, qubits=[1])
    return rxx_circ