from __future__ import annotations
from typing import Any, List
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers.json import parse_and_check_json_markdown
from langchain_core.pydantic_v1 import BaseModel
from langchain.output_parsers.format_instructions import (
class ResponseSchema(BaseModel):
    """A schema for a response from a structured output parser."""
    name: str
    'The name of the schema.'
    description: str
    'The description of the schema.'
    type: str = 'string'
    'The type of the response.'