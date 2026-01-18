from typing import List, cast, Optional, Union, Dict, Any
import functools
from math import sqrt
import httpx
import numpy as np
import networkx as nx
import cirq
from pyquil.quantum_processor import QCSQuantumProcessor
from qcs_api_client.models import InstructionSetArchitecture
from qcs_api_client.operations.sync import get_instruction_set_architecture
from cirq_rigetti._qcs_api_client_decorator import _provide_default_client
@functools.lru_cache(maxsize=2)
def _line_qubit_mapping(self) -> List[int]:
    mapping: List[int] = []
    for i in range(self._number_octagons):
        base = i * 10
        mapping = mapping + [base + index for index in _forward_line_qubit_mapping]
    for i in range(self._number_octagons):
        base = (self._number_octagons - i - 1) * 10
        mapping = mapping + [base + index for index in _reverse_line_qubit_mapping]
    return mapping