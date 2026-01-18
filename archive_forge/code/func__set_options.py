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
def _set_options(self, qobj_config: QasmQobjConfig | None=None, backend_options: dict | None=None) -> None:
    """Set the backend options for all experiments in a qobj"""
    self._initial_statevector = self.options.get('initial_statevector')
    self._chop_threshold = self.options.get('chop_threshold')
    if 'backend_options' in backend_options and backend_options['backend_options']:
        backend_options = backend_options['backend_options']
    if 'initial_statevector' in backend_options and backend_options['initial_statevector'] is not None:
        self._initial_statevector = np.array(backend_options['initial_statevector'], dtype=complex)
    elif hasattr(qobj_config, 'initial_statevector'):
        self._initial_statevector = np.array(qobj_config.initial_statevector, dtype=complex)
    if self._initial_statevector is not None:
        norm = np.linalg.norm(self._initial_statevector)
        if round(norm, 12) != 1:
            raise BasicProviderError(f'initial statevector is not normalized: norm {norm} != 1')
    if 'chop_threshold' in backend_options:
        self._chop_threshold = backend_options['chop_threshold']
    elif hasattr(qobj_config, 'chop_threshold'):
        self._chop_threshold = qobj_config.chop_threshold