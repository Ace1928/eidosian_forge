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
class PydanticToolsParser(JsonOutputToolsParser):
    """Parse tools from OpenAI response."""
    tools: List[Type[BaseModel]]

    def parse_result(self, result: List[Generation], *, partial: bool=False) -> Any:
        json_results = super().parse_result(result, partial=partial)
        if not json_results:
            return None if self.first_tool_only else []
        json_results = [json_results] if self.first_tool_only else json_results
        name_dict = {tool.__name__: tool for tool in self.tools}
        pydantic_objects = []
        for res in json_results:
            try:
                if not isinstance(res['args'], dict):
                    raise ValueError(f'Tool arguments must be specified as a dict, received: {res['args']}')
                pydantic_objects.append(name_dict[res['type']](**res['args']))
            except (ValidationError, ValueError) as e:
                if partial:
                    continue
                else:
                    raise e
        if self.first_tool_only:
            return pydantic_objects[0] if pydantic_objects else None
        else:
            return pydantic_objects