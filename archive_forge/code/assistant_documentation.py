from typing import List, Optional
from typing_extensions import Literal
from ..._models import BaseModel
from .assistant_tool import AssistantTool
A list of tool enabled on the assistant.

    There can be a maximum of 128 tools per assistant. Tools can be of types
    `code_interpreter`, `retrieval`, or `function`.
    