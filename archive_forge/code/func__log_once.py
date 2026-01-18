import functools
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import Field
from langchain.chains.base import Chain
@staticmethod
@functools.lru_cache
def _log_once(msg: str) -> None:
    """Log a message once.

        :meta private:
        """
    logger.warning(msg)