import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.library.basis_change import QFT
from .adder import Adder

        Args:
            num_state_qubits: The number of qubits in either input register for
                state :math:`|a\rangle` or :math:`|b\rangle`. The two input
                registers must have the same number of qubits.
            kind: The kind of adder, can be ``'half'`` for a half adder or
                ``'fixed'`` for a fixed-sized adder. A half adder contains a carry-out to represent
                the most-significant bit, but the fixed-sized adder doesn't and hence performs
                addition modulo ``2 ** num_state_qubits``.
            name: The name of the circuit object.
        Raises:
            ValueError: If ``num_state_qubits`` is lower than 1.
        