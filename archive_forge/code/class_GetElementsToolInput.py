from __future__ import annotations
import json
from typing import TYPE_CHECKING, List, Optional, Sequence, Type
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.playwright.base import BaseBrowserTool
from langchain_community.tools.playwright.utils import (
class GetElementsToolInput(BaseModel):
    """Input for GetElementsTool."""
    selector: str = Field(..., description="CSS selector, such as '*', 'div', 'p', 'a', #id, .classname")
    attributes: List[str] = Field(default_factory=lambda: ['innerText'], description='Set of attributes to retrieve for each element')