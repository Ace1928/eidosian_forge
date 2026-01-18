import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class ExecutorJob(Message):
    """
    Job which is sent directly to the executor
    """
    instrument_settings: Dict[str, Any]
    'Dict mapping instrument names to arbitrary instrument\n          settings.'
    filter_pipeline: Dict[str, FilterNode]
    'The filter pipeline to process measured data.'
    receivers: Dict[str, Receiver]
    'Dict mapping stream names to receiver settings.'
    duration: Optional[float] = None
    'The total duration of the program execution in seconds.'
    timebomb: Optional[TimeBomb] = None
    'An optional payload used to match this job with a\n          particular execution target.'