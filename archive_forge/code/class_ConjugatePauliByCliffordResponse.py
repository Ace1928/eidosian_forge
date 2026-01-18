from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, List
import rpcq
from qcs_api_client.client import QCSClientConfiguration
from rpcq.messages import TargetDevice as TargetQuantumProcessor
@dataclass
class ConjugatePauliByCliffordResponse:
    """
    Conjugate Pauli by Clifford response.
    """
    phase_factor: int
    'Encoded global phase factor on the emitted Pauli.'
    pauli: str
    'Description of the encoded Pauli.'