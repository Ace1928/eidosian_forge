from typing import Union
from typing_extensions import Literal, Annotated
from .thread import Thread
from ..shared import ErrorObject
from .threads import Run, Message, MessageDeltaEvent
from ..._utils import PropertyInfo
from ..._models import BaseModel
from .threads.runs import RunStep, RunStepDeltaEvent
class ThreadRunStepCancelled(BaseModel):
    data: RunStep
    'Represents a step in execution of a run.'
    event: Literal['thread.run.step.cancelled']