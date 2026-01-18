from __future__ import annotations
import asyncio
from typing import Optional, Dict, Any, List, Union, Type, TYPE_CHECKING
from .base import BaseGlobalClient
@property
def acontext_manager(self) -> 'AsyncPlaywrightContextManager':
    """
        Returns the Playwright context manager
        """
    if self._apw_context_manager is None:
        from playwright.async_api import async_playwright
        self._apw_context_manager = async_playwright()
    return self._apw_context_manager