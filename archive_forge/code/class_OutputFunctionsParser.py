import copy
import json
from typing import Any, Dict, List, Optional, Type, Union
import jsonpatch
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import (
from langchain_core.output_parsers.json import parse_partial_json
from langchain_core.outputs.chat_generation import (
from langchain_core.pydantic_v1 import BaseModel, root_validator
class OutputFunctionsParser(BaseGenerationOutputParser[Any]):
    """Parse an output that is one of sets of values."""
    args_only: bool = True
    'Whether to only return the arguments to the function call.'

    def parse_result(self, result: List[Generation], *, partial: bool=False) -> Any:
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            raise OutputParserException('This output parser can only be used with a chat generation.')
        message = generation.message
        try:
            func_call = copy.deepcopy(message.additional_kwargs['function_call'])
        except KeyError as exc:
            raise OutputParserException(f'Could not parse function call: {exc}')
        if self.args_only:
            return func_call['arguments']
        return func_call