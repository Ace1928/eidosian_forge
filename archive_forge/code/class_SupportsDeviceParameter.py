from typing import Any, Dict, Optional, Sequence
import dataclasses
from typing_extensions import Protocol
import cirq
class SupportsDeviceParameter(Protocol):
    """Protocol for using device parameter keys.

    Args:
       path: path of the key to modify, with each sub-folder as a string
           entry in a list.
       idx: If this key is an array, which index to modify.
       value: value of the parameter to be set, if any.
    """
    path: Sequence[str]
    idx: Optional[int] = None
    value: Optional[Any] = None