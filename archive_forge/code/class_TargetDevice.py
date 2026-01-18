import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
@dataclass(eq=False, repr=False)
class TargetDevice(Message):
    """
    ISA and specs for a particular device.
    """
    isa: Dict[str, Dict]
    'Instruction-set architecture for this device.'
    specs: Dict[str, Dict]
    'Fidelities and coherence times for this device.'