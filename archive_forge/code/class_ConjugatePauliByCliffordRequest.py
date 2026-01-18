from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, List
import rpcq
from qcs_api_client.client import QCSClientConfiguration
from rpcq.messages import TargetDevice as TargetQuantumProcessor
@dataclass
class ConjugatePauliByCliffordRequest:
    """
    Request to conjugate a Pauli element by a Clifford element.
    """
    pauli_indices: List[int]
    'Qubit indices onto which the factors of the Pauli term are applied.'
    pauli_symbols: List[str]
    'Ordered factors of the Pauli term.'
    clifford: str
    'Clifford element.'