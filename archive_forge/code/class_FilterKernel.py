import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class FilterKernel(AbstractKernel):
    """
    A filter kernel to produce scalar readout features from acquired readout waveforms.
    """
    iqs: List[float] = field(default_factory=list)
    'The raw kernel coefficients, alternating real and imaginary parts.'
    bias: float = 0.0
    'The kernel is offset by this real value. Can be used to ensure the decision threshold lies at 0.0.'