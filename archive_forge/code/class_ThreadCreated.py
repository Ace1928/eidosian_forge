from typing import Union
from typing_extensions import Literal, Annotated
from .thread import Thread
from ..shared import ErrorObject
from .threads import Run, Message, MessageDeltaEvent
from ..._utils import PropertyInfo
from ..._models import BaseModel
from .threads.runs import RunStep, RunStepDeltaEvent
class ThreadCreated(BaseModel):
    data: Thread
    '\n    Represents a thread that contains\n    [messages](https://platform.openai.com/docs/api-reference/messages).\n    '
    event: Literal['thread.created']