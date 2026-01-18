import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class QGSChannel(Message):
    """
    Configuration for a single QGS Channel.
    """
    direction: Optional[str] = 'tx'
    'The QGS is a device that transmits pulses.'
    nco_frequency: Optional[float] = 2000000000.0
    'The DAC NCO frequency [Hz].'
    gain: Optional[float] = 0.0
    'The output gain on the DAC in [dB]. Note that this\n          should be in the range -45dB to 0dB and is rounded to the\n          nearest 3dB step.'
    channel_index: Optional[int] = 0
    'The channel index on the QGS, zero indexed from the lowest channel,\n        as installed in the box.'
    delay: float = 0.0
    'Delay [seconds] to account for inter-channel skew.'