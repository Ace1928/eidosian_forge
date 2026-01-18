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
def _aspen_qubit_index(self, valid_qubit: cirq.Qid) -> int:
    if isinstance(valid_qubit, cirq.GridQubit):
        return _grid_qubit_mapping[valid_qubit]
    if isinstance(valid_qubit, cirq.LineQubit):
        return self._line_qubit_mapping()[valid_qubit.x]
    if isinstance(valid_qubit, cirq.NamedQubit):
        return int(valid_qubit.name)
    if isinstance(valid_qubit, (OctagonalQubit, AspenQubit)):
        return valid_qubit.index
    else:
        raise UnsupportedQubit(f'unsupported Qid type {type(valid_qubit)}')