import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class DeployedRack(Message):
    """
    The rack configuration for lodgepole.
    """
    rack_meta: RackMeta
    'Meta information about the deployed rack.'
    qpu: QPU
    'Information about the QPU.'
    instruments: Dict[str, Instrument] = field(default_factory=dict)
    'Mapping of instrument name to instrument settings.'