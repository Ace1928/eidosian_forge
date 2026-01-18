import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class QGSx2Sequencer(Message):
    """
    Configuration for a single QGSx2 Sequencer.
    """
    tx_channel: str
    'The label of the associated channel.'
    sequencer_index: int
    'The sequencer index of this sequencer.'