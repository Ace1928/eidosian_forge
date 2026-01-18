from typing import List, Optional
from typing_extensions import Literal
from ...._models import BaseModel
from .run_status import RunStatus
from ..assistant_tool import AssistantTool
from .required_action_function_tool_call import RequiredActionFunctionToolCall
class RequiredAction(BaseModel):
    submit_tool_outputs: RequiredActionSubmitToolOutputs
    'Details on the tool outputs needed for this run to continue.'
    type: Literal['submit_tool_outputs']
    'For now, this is always `submit_tool_outputs`.'