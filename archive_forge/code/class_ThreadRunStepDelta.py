from typing import Union
from typing_extensions import Literal, Annotated
from .thread import Thread
from ..shared import ErrorObject
from .threads import Run, Message, MessageDeltaEvent
from ..._utils import PropertyInfo
from ..._models import BaseModel
from .threads.runs import RunStep, RunStepDeltaEvent
class ThreadRunStepDelta(BaseModel):
    data: RunStepDeltaEvent
    'Represents a run step delta i.e.\n\n    any changed fields on a run step during streaming.\n    '
    event: Literal['thread.run.step.delta']