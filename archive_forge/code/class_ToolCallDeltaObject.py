from typing import List, Optional
from typing_extensions import Literal
from ....._models import BaseModel
from .tool_call_delta import ToolCallDelta
class ToolCallDeltaObject(BaseModel):
    type: Literal['tool_calls']
    'Always `tool_calls`.'
    tool_calls: Optional[List[ToolCallDelta]] = None
    'An array of tool calls the run step was involved in.\n\n    These can be associated with one of three types of tools: `code_interpreter`,\n    `retrieval`, or `function`.\n    '