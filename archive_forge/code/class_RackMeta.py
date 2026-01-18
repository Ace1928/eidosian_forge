import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class RackMeta(Message):
    """
    Meta information about a rack configuration.
    """
    rack_id: Optional[str] = None
    'A unique identifier for the rack.'
    rack_version: Optional[int] = None
    'A version of the rack configuration.'
    schema_version: int = 0
    'A version of the rack configuration.'