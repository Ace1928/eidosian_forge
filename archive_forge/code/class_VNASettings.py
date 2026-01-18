import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class VNASettings(Message):
    """
    Configuration of VNA settings for a continuous wave sweep.
    """
    e_delay: float
    'Electrical delay in seconds from source to measure port'
    phase_offset: float
    'Phase offset in degrees from measured to reported phase'
    bandwidth: float
    'Bandwidth of the sweep, in Hz'
    power: float
    'Source power in dBm'
    freq_sweep: CWFrequencySweep
    'Frequency sweep settings'
    averaging: int = 1
    'Sets the number of points to combine into an averaged\n          trace'