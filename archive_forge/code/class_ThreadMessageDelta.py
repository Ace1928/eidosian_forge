from typing import Union
from typing_extensions import Literal, Annotated
from .thread import Thread
from ..shared import ErrorObject
from .threads import Run, Message, MessageDeltaEvent
from ..._utils import PropertyInfo
from ..._models import BaseModel
from .threads.runs import RunStep, RunStepDeltaEvent
class ThreadMessageDelta(BaseModel):
    data: MessageDeltaEvent
    'Represents a message delta i.e.\n\n    any changed fields on a message during streaming.\n    '
    event: Literal['thread.message.delta']