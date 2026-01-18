from typing import Optional
from typing_extensions import Literal
from ....._models import BaseModel
class FunctionToolCall(BaseModel):
    id: str
    'The ID of the tool call object.'
    function: Function
    'The definition of the function that was called.'
    type: Literal['function']
    'The type of tool call.\n\n    This is always going to be `function` for this type of tool call.\n    '