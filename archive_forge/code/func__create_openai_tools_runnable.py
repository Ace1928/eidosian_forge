import json
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Type, Union
from langchain_core.output_parsers import (
from langchain_core.output_parsers.openai_functions import (
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import (
from langchain.output_parsers import (
def _create_openai_tools_runnable(tool: Union[Dict[str, Any], Type[BaseModel], Callable], llm: Runnable, *, prompt: Optional[BasePromptTemplate], output_parser: Optional[Union[BaseOutputParser, BaseGenerationOutputParser]], enforce_tool_usage: bool, first_tool_only: bool) -> Runnable:
    oai_tool = convert_to_openai_tool(tool)
    llm_kwargs: Dict[str, Any] = {'tools': [oai_tool]}
    if enforce_tool_usage:
        llm_kwargs['tool_choice'] = {'type': 'function', 'function': {'name': oai_tool['function']['name']}}
    output_parser = output_parser or _get_openai_tool_output_parser(tool, first_tool_only=first_tool_only)
    if prompt:
        return prompt | llm.bind(**llm_kwargs) | output_parser
    else:
        return llm.bind(**llm_kwargs) | output_parser