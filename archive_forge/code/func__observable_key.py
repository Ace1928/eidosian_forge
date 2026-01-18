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
def _observable_key(observable: SparsePauliOp) -> tuple:
    """Private key function for SparsePauliOp.
    Args:
        observable: Input operator.

    Returns:
        Key for observables.
    """
    return (observable.paulis.z.tobytes(), observable.paulis.x.tobytes(), observable.paulis.phase.tobytes(), observable.coeffs.tobytes())