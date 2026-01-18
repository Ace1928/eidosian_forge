import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
@dataclass(eq=False, repr=False)
class NativeQuilMetadata(Message):
    """
    Metadata for a native quil program.
    """
    final_rewiring: List[int] = field(default_factory=list)
    'Output qubit index relabeling due to SWAP insertion.'
    gate_depth: Optional[int] = None
    'Maximum number of successive gates in the native quil program.'
    gate_volume: Optional[int] = None
    'Total number of gates in the native quil program.'
    multiqubit_gate_depth: Optional[int] = None
    'Maximum number of successive two-qubit gates in the native quil program.'
    program_duration: Optional[float] = None
    'Rough estimate of native quil program length in nanoseconds.'
    program_fidelity: Optional[float] = None
    'Rough estimate of the fidelity of the full native quil program, uses specs.'
    topological_swaps: Optional[int] = None
    'Total number of SWAPs in the native quil program.'
    qpu_runtime_estimation: Optional[float] = None
    'The estimated runtime (milliseconds) on a Rigetti QPU for a protoquil program.'