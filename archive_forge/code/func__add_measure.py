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
def _add_measure(self, qubit: int, cmembit: int, cregbit: int | None=None) -> None:
    """Apply a measure instruction to a qubit.

        Args:
            qubit: qubit is the qubit measured.
            cmembit: is the classical memory bit to store outcome in.
            cregbit: is the classical register bit to store outcome in.
        """
    outcome, probability = self._get_measure_outcome(qubit)
    membit = 1 << cmembit
    self._classical_memory = self._classical_memory & ~membit | int(outcome) << cmembit
    if cregbit is not None:
        regbit = 1 << cregbit
        self._classical_register = self._classical_register & ~regbit | int(outcome) << cregbit
    if outcome == '0':
        update_diag = [[1 / np.sqrt(probability), 0], [0, 0]]
    else:
        update_diag = [[0, 0], [0, 1 / np.sqrt(probability)]]
    self._add_unitary(update_diag, [qubit])