from __future__ import annotations
import asyncio
from typing import Optional, Dict, Any, List, Union, Type, TYPE_CHECKING
from .base import BaseGlobalClient
@property
def aclient(self) -> 'AsyncPlaywright':
    """
        Returns the Playwright instance
        """
    if self._apw_client is None:
        raise RuntimeError('AsyncPlaywright not initialized. Initialize with Browser.ainit()')
    return self._apw_client