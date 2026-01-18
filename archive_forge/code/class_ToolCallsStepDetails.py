from typing import List
from typing_extensions import Literal
from .tool_call import ToolCall
from ....._models import BaseModel
class ToolCallsStepDetails(BaseModel):
    tool_calls: List[ToolCall]
    'An array of tool calls the run step was involved in.\n\n    These can be associated with one of three types of tools: `code_interpreter`,\n    `retrieval`, or `function`.\n    '
    type: Literal['tool_calls']
    'Always `tool_calls`.'