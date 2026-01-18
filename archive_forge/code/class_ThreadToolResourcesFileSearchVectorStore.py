from __future__ import annotations
from typing import List, Union, Iterable, Optional
from typing_extensions import Literal, Required, TypedDict
from .function_tool_param import FunctionToolParam
from .file_search_tool_param import FileSearchToolParam
from .code_interpreter_tool_param import CodeInterpreterToolParam
from .assistant_tool_choice_option_param import AssistantToolChoiceOptionParam
from .assistant_response_format_option_param import AssistantResponseFormatOptionParam
class ThreadToolResourcesFileSearchVectorStore(TypedDict, total=False):
    file_ids: List[str]
    '\n    A list of [file](https://platform.openai.com/docs/api-reference/files) IDs to\n    add to the vector store. There can be a maximum of 10000 files in a vector\n    store.\n    '
    metadata: object
    'Set of 16 key-value pairs that can be attached to a vector store.\n\n    This can be useful for storing additional information about the vector store in\n    a structured format. Keys can be a maximum of 64 characters long and values can\n    be a maxium of 512 characters long.\n    '