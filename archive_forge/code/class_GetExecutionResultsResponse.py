import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
@dataclass(eq=False, repr=False)
class GetExecutionResultsResponse(Message):
    """
    Results of a completed ExecutorJob execution.
    """
    buffers: Dict[str, Dict[str, Any]]
    'Result buffers for a completed ExecutorJob.'
    execution_duration_microseconds: int
    'Duration (in microseconds) ExecutorJob held exclusive access to quantum hardware.'