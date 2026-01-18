import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class DebugMessage(Instruction):
    """
    Instructs the target to emit a specified debug message.
    """
    frame: str
    'The frame label that owns this debug message.'
    message: int
    'The 2-byte wide debug message to emit.'