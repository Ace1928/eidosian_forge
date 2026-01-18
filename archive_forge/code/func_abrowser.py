from __future__ import annotations
import asyncio
from typing import Optional, Dict, Any, List, Union, Type, TYPE_CHECKING
from .base import BaseGlobalClient
@property
def abrowser(self) -> 'AsyncBrowser':
    """
        Returns the browser instance
        """
    if self._abrowser is None:
        raise RuntimeError('AsyncBrowser not initialized. Initialize with Browser.ainit()')
    return self._abrowser