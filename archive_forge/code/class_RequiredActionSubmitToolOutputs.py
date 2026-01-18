from typing import List, Optional
from typing_extensions import Literal
from ...._models import BaseModel
from .run_status import RunStatus
from ..assistant_tool import AssistantTool
from .required_action_function_tool_call import RequiredActionFunctionToolCall
class RequiredActionSubmitToolOutputs(BaseModel):
    tool_calls: List[RequiredActionFunctionToolCall]
    'A list of the relevant tool calls.'