from typing import List, Optional
from typing_extensions import Literal
from ...._models import BaseModel
from .run_status import RunStatus
from ..assistant_tool import AssistantTool
from .required_action_function_tool_call import RequiredActionFunctionToolCall
class Run(BaseModel):
    id: str
    'The identifier, which can be referenced in API endpoints.'
    assistant_id: str
    '\n    The ID of the\n    [assistant](https://platform.openai.com/docs/api-reference/assistants) used for\n    execution of this run.\n    '
    cancelled_at: Optional[int] = None
    'The Unix timestamp (in seconds) for when the run was cancelled.'
    completed_at: Optional[int] = None
    'The Unix timestamp (in seconds) for when the run was completed.'
    created_at: int
    'The Unix timestamp (in seconds) for when the run was created.'
    expires_at: Optional[int] = None
    'The Unix timestamp (in seconds) for when the run will expire.'
    failed_at: Optional[int] = None
    'The Unix timestamp (in seconds) for when the run failed.'
    file_ids: List[str]
    '\n    The list of [File](https://platform.openai.com/docs/api-reference/files) IDs the\n    [assistant](https://platform.openai.com/docs/api-reference/assistants) used for\n    this run.\n    '
    instructions: str
    '\n    The instructions that the\n    [assistant](https://platform.openai.com/docs/api-reference/assistants) used for\n    this run.\n    '
    last_error: Optional[LastError] = None
    'The last error associated with this run. Will be `null` if there are no errors.'
    metadata: Optional[object] = None
    'Set of 16 key-value pairs that can be attached to an object.\n\n    This can be useful for storing additional information about the object in a\n    structured format. Keys can be a maximum of 64 characters long and values can be\n    a maxium of 512 characters long.\n    '
    model: str
    '\n    The model that the\n    [assistant](https://platform.openai.com/docs/api-reference/assistants) used for\n    this run.\n    '
    object: Literal['thread.run']
    'The object type, which is always `thread.run`.'
    required_action: Optional[RequiredAction] = None
    'Details on the action required to continue the run.\n\n    Will be `null` if no action is required.\n    '
    started_at: Optional[int] = None
    'The Unix timestamp (in seconds) for when the run was started.'
    status: RunStatus
    '\n    The status of the run, which can be either `queued`, `in_progress`,\n    `requires_action`, `cancelling`, `cancelled`, `failed`, `completed`, or\n    `expired`.\n    '
    thread_id: str
    '\n    The ID of the [thread](https://platform.openai.com/docs/api-reference/threads)\n    that was executed on as a part of this run.\n    '
    tools: List[AssistantTool]
    '\n    The list of tools that the\n    [assistant](https://platform.openai.com/docs/api-reference/assistants) used for\n    this run.\n    '
    usage: Optional[Usage] = None
    'Usage statistics related to the run.\n\n    This value will be `null` if the run is not in a terminal state (i.e.\n    `in_progress`, `queued`, etc.).\n    '