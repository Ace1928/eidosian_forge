from __future__ import annotations
import copy
from typing import Literal, TYPE_CHECKING
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.barrier import Barrier
from qiskit.circuit.delay import Delay
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.mixins import AdjointMixin, MultiplyMixin
def _evolve_cx(base_pauli, qctrl, qtrgt):
    """Update P -> CX.P.CX"""
    base_pauli._x[:, qtrgt] ^= base_pauli._x[:, qctrl]
    base_pauli._z[:, qctrl] ^= base_pauli._z[:, qtrgt]
    return base_pauli