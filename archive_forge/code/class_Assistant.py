from typing import List, Optional
from typing_extensions import Literal
from ..._models import BaseModel
from .assistant_tool import AssistantTool
class Assistant(BaseModel):
    id: str
    'The identifier, which can be referenced in API endpoints.'
    created_at: int
    'The Unix timestamp (in seconds) for when the assistant was created.'
    description: Optional[str] = None
    'The description of the assistant. The maximum length is 512 characters.'
    file_ids: List[str]
    '\n    A list of [file](https://platform.openai.com/docs/api-reference/files) IDs\n    attached to this assistant. There can be a maximum of 20 files attached to the\n    assistant. Files are ordered by their creation date in ascending order.\n    '
    instructions: Optional[str] = None
    'The system instructions that the assistant uses.\n\n    The maximum length is 32768 characters.\n    '
    metadata: Optional[object] = None
    'Set of 16 key-value pairs that can be attached to an object.\n\n    This can be useful for storing additional information about the object in a\n    structured format. Keys can be a maximum of 64 characters long and values can be\n    a maxium of 512 characters long.\n    '
    model: str
    'ID of the model to use.\n\n    You can use the\n    [List models](https://platform.openai.com/docs/api-reference/models/list) API to\n    see all of your available models, or see our\n    [Model overview](https://platform.openai.com/docs/models/overview) for\n    descriptions of them.\n    '
    name: Optional[str] = None
    'The name of the assistant. The maximum length is 256 characters.'
    object: Literal['assistant']
    'The object type, which is always `assistant`.'
    tools: List[AssistantTool]
    'A list of tool enabled on the assistant.\n\n    There can be a maximum of 128 tools per assistant. Tools can be of types\n    `code_interpreter`, `retrieval`, or `function`.\n    '