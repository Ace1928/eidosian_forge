from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, List
import rpcq
from qcs_api_client.client import QCSClientConfiguration
from rpcq.messages import TargetDevice as TargetQuantumProcessor
@dataclass
class NativeQuilMetadataResponse:
    """
    Metadata for a native Quil program.
    """
    final_rewiring: List[int]
    'Output qubit index relabeling due to SWAP insertion.'
    gate_depth: Optional[int]
    'Maximum number of successive gates in the native Quil program.'
    gate_volume: Optional[int]
    'Total number of gates in the native Quil program.'
    multiqubit_gate_depth: Optional[int]
    'Maximum number of successive two-qubit gates in the native Quil program.'
    program_duration: Optional[float]
    'Rough estimate of native Quil program length in nanoseconds.'
    program_fidelity: Optional[float]
    'Rough estimate of the fidelity of the full native Quil program.'
    topological_swaps: Optional[int]
    'Total number of SWAPs in the native Quil program.'
    qpu_runtime_estimation: Optional[float]
    '\n    The estimated runtime (milliseconds) on a Rigetti QPU (protoquil program). Available only for protoquil-compliant\n    programs.\n    '