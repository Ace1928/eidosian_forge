from __future__ import annotations
import asyncio
from typing import Optional, Dict, Any, List, Union, Type, TYPE_CHECKING
from .base import BaseGlobalClient
def _set_abrowser(self, task: asyncio.Task):
    """
        Sets the AsyncBrowser instance
        """
    self.logger.info('Setting AsyncBrowser')
    self._abrowser = task.result()