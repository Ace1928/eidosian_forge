import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class QDOSlowFluxChannel(Message):
    """
    Configuration for a single QDO Slow Flux Channel.
    """
    channel_index: int
    'The channel index on the QDO, zero indexed from the\n          lowest channel, as installed in the box. Flux index typically starts at 4.'
    flux_current: Optional[float] = None
    'Flux current [Amps].'
    relay_closed: Optional[bool] = False
    'Set the state of the Flux relay.\n          True  - Relay closed, allows flux current to flow.\n          False - Relay open, no flux current can flow.'