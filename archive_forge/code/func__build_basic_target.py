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
def _build_basic_target(self) -> Target:
    """Helper method that returns a minimal target with a basis gate set but
        no coupling map, instruction properties or calibrations.

        Returns:
            The configured target.
        """
    target = Target(description='Basic Target', num_qubits=None)
    basis_gates = ['h', 'u', 'p', 'u1', 'u2', 'u3', 'rz', 'sx', 'x', 'cx', 'id', 'unitary', 'measure', 'delay', 'reset']
    inst_mapping = get_standard_gate_name_mapping()
    for name in basis_gates:
        if name in inst_mapping:
            instruction = inst_mapping[name]
            target.add_instruction(instruction, properties=None, name=name)
        elif name == 'unitary':
            target.add_instruction(UnitaryGate, name='unitary')
        else:
            raise BasicProviderError('Gate is not a valid basis gate for this simulator: %s' % name)
    return target