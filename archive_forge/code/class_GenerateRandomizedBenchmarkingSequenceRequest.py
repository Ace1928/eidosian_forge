from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, List
import rpcq
from qcs_api_client.client import QCSClientConfiguration
from rpcq.messages import TargetDevice as TargetQuantumProcessor
@dataclass
class GenerateRandomizedBenchmarkingSequenceRequest:
    """
    Request to generate a randomized benchmarking sequence.
    """
    depth: int
    'Depth of the benchmarking sequence.'
    num_qubits: int
    'Number of qubits involved in the benchmarking sequence.'
    gateset: List[str]
    'List of Quil programs, each describing a Clifford.'
    seed: Optional[int]
    'PRNG seed. Set this to guarantee repeatable results.'
    interleaver: Optional[str]
    'Fixed Clifford, specified as a Quil string, to interleave through an RB sequence.'