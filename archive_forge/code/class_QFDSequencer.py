import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class QFDSequencer(Message):
    """
    Configuration for a single QFD Sequencer.
    """
    tx_channel: str
    'The label of the associated channel.'
    sequencer_index: int
    'The sequencer index of this sequencer.'