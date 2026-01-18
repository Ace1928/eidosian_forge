from __future__ import annotations
import sys as _sys
from typing import Any as _Any
from ._pydantic_core import (
from .core_schema import CoreConfig, CoreSchema, CoreSchemaType, ErrorType
class ErrorTypeInfo(_TypedDict):
    """
    Gives information about errors.
    """
    type: ErrorType
    'The type of error that occurred, this should a "slug" identifier that changes rarely or never.'
    message_template_python: str
    'String template to render a human readable error message from using context, when the input is Python.'
    example_message_python: str
    'Example of a human readable error message, when the input is Python.'
    message_template_json: _NotRequired[str]
    'String template to render a human readable error message from using context, when the input is JSON data.'
    example_message_json: _NotRequired[str]
    'Example of a human readable error message, when the input is JSON data.'
    example_context: dict[str, _Any] | None
    'Example of context values.'