import json
from typing import Generic, List, Type, TypeVar, Union
import pydantic  # pydantic: ignore
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import Generation
from langchain_core.utils.pydantic import PYDANTIC_MAJOR_VERSION
def _parser_exception(self, e: Exception, json_object: dict) -> OutputParserException:
    json_string = json.dumps(json_object)
    name = self.pydantic_object.__name__
    msg = f'Failed to parse {name} from completion {json_string}. Got: {e}'
    return OutputParserException(msg, llm_output=json_string)