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
def _add_sample_measure(self, measure_params: list[list[int, int]], num_samples: int) -> list[hex]:
    """Generate memory samples from current statevector.

        Args:
            measure_params: List of (qubit, cmembit) values for
                                   measure instructions to sample.
            num_samples: The number of memory samples to generate.

        Returns:
            A list of memory values in hex format.
        """
    measured_qubits = sorted({qubit for qubit, cmembit in measure_params})
    num_measured = len(measured_qubits)
    axis = list(range(self._number_of_qubits))
    for qubit in reversed(measured_qubits):
        axis.remove(self._number_of_qubits - 1 - qubit)
    probabilities = np.reshape(np.sum(np.abs(self._statevector) ** 2, axis=tuple(axis)), 2 ** num_measured)
    samples = self._local_random.choice(range(2 ** num_measured), num_samples, p=probabilities)
    memory = []
    for sample in samples:
        classical_memory = self._classical_memory
        for qubit, cmembit in measure_params:
            pos = measured_qubits.index(qubit)
            qubit_outcome = int((sample & 1 << pos) >> pos)
            membit = 1 << cmembit
            classical_memory = classical_memory & ~membit | qubit_outcome << cmembit
        value = bin(classical_memory)[2:]
        memory.append(hex(int(value, 2)))
    return memory