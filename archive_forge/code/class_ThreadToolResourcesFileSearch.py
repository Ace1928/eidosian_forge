from __future__ import annotations
from typing import List, Union, Iterable, Optional
from typing_extensions import Literal, Required, TypedDict
from .function_tool_param import FunctionToolParam
from .file_search_tool_param import FileSearchToolParam
from .code_interpreter_tool_param import CodeInterpreterToolParam
from .assistant_tool_choice_option_param import AssistantToolChoiceOptionParam
from .assistant_response_format_option_param import AssistantResponseFormatOptionParam
class ThreadToolResourcesFileSearch(TypedDict, total=False):
    vector_store_ids: List[str]
    '\n    The\n    [vector store](https://platform.openai.com/docs/api-reference/vector-stores/object)\n    attached to this thread. There can be a maximum of 1 vector store attached to\n    the thread.\n    '
    vector_stores: Iterable[ThreadToolResourcesFileSearchVectorStore]
    '\n    A helper to create a\n    [vector store](https://platform.openai.com/docs/api-reference/vector-stores/object)\n    with file_ids and attach it to this thread. There can be a maximum of 1 vector\n    store attached to the thread.\n    '