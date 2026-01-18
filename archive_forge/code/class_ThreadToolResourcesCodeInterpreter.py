from __future__ import annotations
from typing import List, Union, Iterable, Optional
from typing_extensions import Literal, Required, TypedDict
from .function_tool_param import FunctionToolParam
from .file_search_tool_param import FileSearchToolParam
from .code_interpreter_tool_param import CodeInterpreterToolParam
from .assistant_tool_choice_option_param import AssistantToolChoiceOptionParam
from .assistant_response_format_option_param import AssistantResponseFormatOptionParam
class ThreadToolResourcesCodeInterpreter(TypedDict, total=False):
    file_ids: List[str]
    '\n    A list of [file](https://platform.openai.com/docs/api-reference/files) IDs made\n    available to the `code_interpreter` tool. There can be a maximum of 20 files\n    associated with the tool.\n    '