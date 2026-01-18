import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class ScheduleIRJob(Message):
    """
    The unit of work to be executed.
    """
    num_shots: int
    'How many repetitions the job should be executed for.'
    resources: Resources
    'The resources required by the job.'
    program: Program
    'The actual program to be executed.'
    operating_point: Dict[str, Dict] = field(default_factory=dict)
    'Operating points or static instrument channel settings\n          (mapping control_name (instrument name) -> instrument channel settings\n          (instrument settings) dictionary).'
    filter_pipeline: Dict[str, FilterNode] = field(default_factory=dict)
    "The filter pipeline. Mapping of node labels to\n          FilterNode's."
    job_id: InitVar[Optional[str]] = None
    'A unique ID to help the submitter track the job.'

    def _extend_by_deprecated_fields(self, d):
        super()._extend_by_deprecated_fields(d)

    def __post_init__(self, job_id):
        if job_id is not None:
            warn("job_id is deprecated; please don't set it anymore")