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
def _get_measure_outcome(self, qubit: int) -> tuple[str, int]:
    """Simulate the outcome of measurement of a qubit.

        Args:
            qubit: the qubit to measure

        Return:
            pair (outcome, probability) where outcome is '0' or '1' and
            probability is the probability of the returned outcome.
        """
    axis = list(range(self._number_of_qubits))
    axis.remove(self._number_of_qubits - 1 - qubit)
    probabilities = np.sum(np.abs(self._statevector) ** 2, axis=tuple(axis))
    random_number = self._local_random.random()
    if random_number < probabilities[0]:
        return ('0', probabilities[0])
    return ('1', probabilities[1])