import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class QRRSequencer(Message):
    """
    Configuration for a single readout receive (QRR) sequencer.
    """
    rx_channel: str
    'The label of the associated rx channel.'
    sequencer_index: int
    'The sequencer index (0-15) to assign. Note that only\n         sequencer 0 can return raw readout measurements.'