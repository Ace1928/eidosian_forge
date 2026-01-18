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
def _circuit_key(circuit: QuantumCircuit, functional: bool=True) -> tuple:
    """Private key function for QuantumCircuit.

    This is the workaround until :meth:`QuantumCircuit.__hash__` will be introduced.
    If key collision is found, please add elements to avoid it.

    Args:
        circuit: Input quantum circuit.
        functional: If True, the returned key only includes functional data (i.e. execution related).

    Returns:
        Composite key for circuit.
    """
    functional_key: tuple = (circuit.num_qubits, circuit.num_clbits, circuit.num_parameters, tuple(((_bits_key(data.qubits, circuit), _bits_key(data.clbits, circuit), data.operation.name, tuple((_format_params(param) for param in data.operation.params))) for data in circuit.data)), None if circuit._op_start_times is None else tuple(circuit._op_start_times))
    if functional:
        return functional_key
    return (circuit.name, *functional_key)