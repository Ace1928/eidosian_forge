from __future__ import annotations
from typing import Optional, Type
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.playwright.base import BaseBrowserTool
from langchain_community.tools.playwright.utils import (
def _selector_effective(self, selector: str) -> str:
    if not self.visible_only:
        return selector
    return f'{selector} >> visible=1'