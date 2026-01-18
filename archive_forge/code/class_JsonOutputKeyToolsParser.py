import copy
import json
from json import JSONDecodeError
from typing import Any, Dict, List, Optional, Type
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, InvalidToolCall
from langchain_core.output_parsers import BaseCumulativeTransformOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.pydantic_v1 import BaseModel, ValidationError
from langchain_core.utils.json import parse_partial_json
class JsonOutputKeyToolsParser(JsonOutputToolsParser):
    """Parse tools from OpenAI response."""
    key_name: str
    'The type of tools to return.'

    def parse_result(self, result: List[Generation], *, partial: bool=False) -> Any:
        parsed_result = super().parse_result(result, partial=partial)
        if self.first_tool_only:
            single_result = parsed_result if parsed_result and parsed_result['type'] == self.key_name else None
            if self.return_id:
                return single_result
            elif single_result:
                return single_result['args']
            else:
                return None
        parsed_result = [res for res in parsed_result if res['type'] == self.key_name]
        if not self.return_id:
            parsed_result = [res['args'] for res in parsed_result]
        return parsed_result