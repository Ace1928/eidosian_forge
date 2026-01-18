from typing import Literal, Optional, Type, TypedDict
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils.json_schema import dereference_refs
class FunctionDescription(TypedDict):
    """Representation of a callable function to the Ernie API."""
    name: str
    'The name of the function.'
    description: str
    'A description of the function.'
    parameters: dict
    'The parameters of the function.'