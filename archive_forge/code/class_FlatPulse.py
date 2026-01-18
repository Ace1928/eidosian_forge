import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class FlatPulse(Instruction):
    """
    Instruction to play a pulse with a constant amplitude
      (except for phase modulation) at a specific time on a specific frame.
    """
    frame: str
    'The tx-frame label on which the pulse is played.'
    iq: List[float]
    'The I and Q value of the constant pulse.'
    duration: float
    "The duration of the pulse in [seconds], should be a\n          multiple of the associated tx-frame's inverse sample rate."
    phase: float = 0.0
    'Static phase angle [units of tau=2pi] by which the\n          envelope quadratures are rotated.'
    detuning: float = 0.0
    'Detuning [Hz] with which the pulse envelope should be\n          modulated relative to the frame frequency.'
    scale: Optional[float] = 1.0
    'Dimensionless (re-)scaling factor which is applied to\n          the envelope.'