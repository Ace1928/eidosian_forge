from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any, Optional, Type
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_community.tools.playwright.base import BaseBrowserTool
from langchain_community.tools.playwright.utils import (
class ExtractHyperlinksToolInput(BaseModel):
    """Input for ExtractHyperlinksTool."""
    absolute_urls: bool = Field(default=False, description='Return absolute URLs instead of relative URLs')