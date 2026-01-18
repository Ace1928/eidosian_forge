from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec
@staticmethod
def _validate_content(content: Optional[Dict[str, MediaType]]) -> None:
    if content:
        raise ValueError("API Properties with media content not supported. Media content only supported within APIRequestBodyProperty's")