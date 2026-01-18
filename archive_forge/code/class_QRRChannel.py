import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class QRRChannel(Message):
    """
    Configuration for a single QRR Channel.
    """
    channel_index: int
    'The channel index on the QRR, zero indexed from the lowest channel,\n        as installed in the box.'
    direction: Optional[str] = 'rx'
    'The QRR is a device that receives readout pulses.'
    nco_frequency: Optional[float] = 0.0
    'The ADC NCO frequency [Hz].'
    gain: Optional[float] = 0.0
    'The input gain on the ADC in [dB]. Note that this should be in the range\n       -45dB to 0dB and is rounded to the nearest 3dB step.'
    delay: float = 0.0
    'Delay [seconds] to account for inter-channel skew.'