from typing_extensions import Literal
from ..._models import BaseModel
class ChatCompletionMessageToolCall(BaseModel):
    id: str
    'The ID of the tool call.'
    function: Function
    'The function that the model called.'
    type: Literal['function']
    'The type of the tool. Currently, only `function` is supported.'