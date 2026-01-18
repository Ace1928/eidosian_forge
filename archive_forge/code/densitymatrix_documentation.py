from __future__ import annotations
import copy
from numbers import Number
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info.operators.mixins.tolerances import TolerancesMixin
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic import Pauli, SparsePauliOp
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import is_positive_semidefinite_matrix
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit._accelerate.pauli_expval import density_expval_pauli_no_x, density_expval_pauli_with_x
from qiskit.quantum_info.states.statevector import Statevector
Return partially transposed density matrix.

        Args:
            qargs (list): The subsystems to be transposed.

        Returns:
            DensityMatrix: The partially transposed density matrix.
        