from __future__ import annotations
import uuid
import time
import logging
import warnings
from collections import Counter
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.providers import Provider
from qiskit.providers.backend import BackendV2
from qiskit.providers.models import BackendConfiguration
from qiskit.providers.options import Options
from qiskit.qobj import QasmQobj, QasmQobjConfig, QasmQobjExperiment
from qiskit.result import Result
from qiskit.transpiler import Target
from .basic_provider_job import BasicProviderJob
from .basic_provider_tools import single_gate_matrix
from .basic_provider_tools import SINGLE_QUBIT_GATES
from .basic_provider_tools import cx_gate_matrix
from .basic_provider_tools import einsum_vecmul_index
from .exceptions import BasicProviderError
def _add_unitary(self, gate: np.ndarray, qubits: list[int]) -> None:
    """Apply an N-qubit unitary matrix.

        Args:
            gate (matrix_like): an N-qubit unitary matrix
            qubits (list): the list of N-qubits.
        """
    num_qubits = len(qubits)
    indexes = einsum_vecmul_index(qubits, self._number_of_qubits)
    gate_tensor = np.reshape(np.array(gate, dtype=complex), num_qubits * [2, 2])
    self._statevector = np.einsum(indexes, gate_tensor, self._statevector, dtype=complex, casting='no')