from __future__ import annotations
import re
from typing import Literal, TYPE_CHECKING
import numpy as np
from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.circuit.barrier import Barrier
from qiskit.circuit.delay import Delay
from qiskit.circuit.library.generalized_gates import PauliGate
from qiskit.circuit.library.standard_gates import IGate, XGate, YGate, ZGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli, _count_y
@classmethod
def _from_pauli_instruction(cls, instr):
    """Convert a Pauli instruction to BasePauli data."""
    if isinstance(instr, PauliGate):
        return cls._from_label(instr.params[0])
    if isinstance(instr, IGate):
        return (np.array([[False]]), np.array([[False]]), np.array([0]))
    if isinstance(instr, XGate):
        return (np.array([[False]]), np.array([[True]]), np.array([0]))
    if isinstance(instr, YGate):
        return (np.array([[True]]), np.array([[True]]), np.array([1]))
    if isinstance(instr, ZGate):
        return (np.array([[True]]), np.array([[False]]), np.array([0]))
    raise QiskitError('Invalid Pauli instruction.')