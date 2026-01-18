import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class QRTx2Sequencer(Message):
    """
    Configuration for a dual readout transmit (QRTx2) sequencer.
    """
    tx_channel: str
    'The label of the associated tx channel.'
    sequencer_index: int
    'The sequencer index (0-15) of this sequencer.'
    low_freq_range: Optional[bool] = False
    'Used to signal if this sequencer is in the low frequency configuration.'