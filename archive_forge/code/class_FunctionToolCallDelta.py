from typing import Optional
from typing_extensions import Literal
from ....._models import BaseModel
class FunctionToolCallDelta(BaseModel):
    index: int
    'The index of the tool call in the tool calls array.'
    type: Literal['function']
    'The type of tool call.\n\n    This is always going to be `function` for this type of tool call.\n    '
    id: Optional[str] = None
    'The ID of the tool call object.'
    function: Optional[Function] = None
    'The definition of the function that was called.'