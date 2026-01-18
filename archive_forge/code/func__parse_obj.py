import json
from typing import Generic, List, Type, TypeVar, Union
import pydantic  # pydantic: ignore
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import Generation
from langchain_core.utils.pydantic import PYDANTIC_MAJOR_VERSION
def _parse_obj(self, obj: dict) -> TBaseModel:
    if PYDANTIC_MAJOR_VERSION == 2:
        try:
            if issubclass(self.pydantic_object, pydantic.BaseModel):
                return self.pydantic_object.model_validate(obj)
            elif issubclass(self.pydantic_object, pydantic.v1.BaseModel):
                return self.pydantic_object.parse_obj(obj)
            else:
                raise OutputParserException(f'Unsupported model version for PydanticOutputParser:                             {self.pydantic_object.__class__}')
        except (pydantic.ValidationError, pydantic.v1.ValidationError) as e:
            raise self._parser_exception(e, obj)
    else:
        try:
            return self.pydantic_object.parse_obj(obj)
        except pydantic.ValidationError as e:
            raise self._parser_exception(e, obj)