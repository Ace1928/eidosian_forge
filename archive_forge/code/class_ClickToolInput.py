from __future__ import annotations
from typing import Optional, Type
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.playwright.base import BaseBrowserTool
from langchain_community.tools.playwright.utils import (
class ClickToolInput(BaseModel):
    """Input for ClickTool."""
    selector: str = Field(..., description='CSS selector for the element to click')