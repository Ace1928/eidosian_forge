import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class DataAxis(Message):
    """
    A data axis allows to label element(s) of a stream.
    """
    name: str
    "Label for the axis, e.g., 'time' or 'shot_index'."
    points: List[float] = field(default_factory=list)
    'The sequence of values along the axis.'