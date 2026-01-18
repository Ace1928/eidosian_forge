from __future__ import annotations
from typing import List, Union, Iterable, Optional
from typing_extensions import Literal, Required, TypedDict
from .function_tool_param import FunctionToolParam
from .retrieval_tool_param import RetrievalToolParam
from .code_interpreter_tool_param import CodeInterpreterToolParam
class ThreadCreateAndRunParamsBase(TypedDict, total=False):
    assistant_id: Required[str]
    '\n    The ID of the\n    [assistant](https://platform.openai.com/docs/api-reference/assistants) to use to\n    execute this run.\n    '
    instructions: Optional[str]
    'Override the default system message of the assistant.\n\n    This is useful for modifying the behavior on a per-run basis.\n    '
    metadata: Optional[object]
    'Set of 16 key-value pairs that can be attached to an object.\n\n    This can be useful for storing additional information about the object in a\n    structured format. Keys can be a maximum of 64 characters long and values can be\n    a maxium of 512 characters long.\n    '
    model: Optional[str]
    '\n    The ID of the [Model](https://platform.openai.com/docs/api-reference/models) to\n    be used to execute this run. If a value is provided here, it will override the\n    model associated with the assistant. If not, the model associated with the\n    assistant will be used.\n    '
    thread: Thread
    'If no thread is provided, an empty thread will be created.'
    tools: Optional[Iterable[Tool]]
    'Override the tools the assistant can use for this run.\n\n    This is useful for modifying the behavior on a per-run basis.\n    '