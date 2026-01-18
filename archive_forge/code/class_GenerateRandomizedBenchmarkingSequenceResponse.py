from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, List
import rpcq
from qcs_api_client.client import QCSClientConfiguration
from rpcq.messages import TargetDevice as TargetQuantumProcessor
@dataclass
class GenerateRandomizedBenchmarkingSequenceResponse:
    """
    Randomly generated benchmarking sequence response.
    """
    sequence: List[List[int]]
    'List of Cliffords, each expressed as a list of generator indices.'