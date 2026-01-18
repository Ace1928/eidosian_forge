import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
@dataclass(eq=False, repr=False)
class NativeQuilRequest(Message):
    """
    Quil and the device metadata necessary for quilc.
    """
    quil: str
    'Arbitrary Quil to be sent to quilc.'
    target_device: TargetDevice
    'Specifications for the device to target with quilc.'