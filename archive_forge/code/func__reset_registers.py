from __future__ import annotations
import numpy as np
from qiskit.circuit import QuantumRegister, AncillaRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from .functional_pauli_rotations import FunctionalPauliRotations
from .linear_pauli_rotations import LinearPauliRotations
from .integer_comparator import IntegerComparator
def _reset_registers(self, num_state_qubits: int | None) -> None:
    """Reset the registers."""
    self.qregs = []
    if num_state_qubits is not None:
        qr_state = QuantumRegister(num_state_qubits)
        qr_target = QuantumRegister(1)
        self.qregs = [qr_state, qr_target]
        if len(self.breakpoints) > 1:
            num_ancillas = num_state_qubits
            qr_ancilla = AncillaRegister(num_ancillas)
            self.add_register(qr_ancilla)