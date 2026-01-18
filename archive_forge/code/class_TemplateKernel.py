import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class TemplateKernel(AbstractKernel):
    """
    An integration kernel defined for a specific frame.
    """
    duration: float
    'Length of the boxcar kernel in seconds'
    bias: float = 0.0
    'The kernel is offset by this real value. Can be used to ensure the decision threshold lies at 0.0.'
    scale: float = 1.0
    'Scale to apply to boxcar kernel'
    phase: float = 0.0
    'Phase [units of tau=2pi] to rotate the kernel by.'
    detuning: float = 0.0
    'Modulation to apply to the filter kernel in Hz'