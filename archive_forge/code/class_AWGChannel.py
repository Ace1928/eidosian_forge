import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class AWGChannel(Message):
    """
    Configuration of a single RF channel.
    """
    sample_rate: float
    'The sampling rate [Hz] of the associated DAC/ADC\n          component.'
    direction: str
    "'rx' or 'tx'"
    lo_frequency: Optional[float] = None
    'The local oscillator frequency [Hz] of the channel.'
    gain: Optional[float] = None
    'If there is an amplifier, the amplifier gain [dB].'
    delay: float = 0.0
    'Delay [seconds] to account for inter-channel skew.'