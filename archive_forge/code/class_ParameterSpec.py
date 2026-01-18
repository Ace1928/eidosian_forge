import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
@dataclass(eq=False, repr=False)
class ParameterSpec(Message):
    """
    Specification of a dynamic parameter type and array-length.
    """
    type: str = ''
    "The parameter type, e.g., one of 'INTEGER', or 'FLOAT'."
    length: int = 1
    'If this is not 1, the parameter is an array of this length.'