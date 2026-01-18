from __future__ import annotations
from collections.abc import Iterable
import numpy as np
from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.circuit.bit import Bit
from qiskit.circuit.library.data_preparation import Initialize
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import PauliList, SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli
def _bits_key(bits: tuple[Bit, ...], circuit: QuantumCircuit) -> tuple:
    return tuple(((circuit.find_bit(bit).index, tuple(((reg[0].size, reg[0].name, reg[1]) for reg in circuit.find_bit(bit).registers))) for bit in bits))