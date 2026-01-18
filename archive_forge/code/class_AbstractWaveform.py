import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class AbstractWaveform(Message):
    """
    A waveform envelope defined for a specific frame. This abstract class is made concrete by either a `Waveform` or a templated waveform such as `GaussianWaveform`
    """
    frame: str
    'The label of the associated tx-frame.'