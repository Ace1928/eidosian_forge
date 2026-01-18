from __future__ import annotations
import logging
import sys
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import (
from uuid import UUID
from tenacity import RetryCallState
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.exceptions import TracerException
from langchain_core.load import dumpd
from langchain_core.messages import BaseMessage
from langchain_core.outputs import (
from langchain_core.tracers.schemas import Run
@staticmethod
def _get_stacktrace(error: BaseException) -> str:
    """Get the stacktrace of the parent error."""
    msg = repr(error)
    try:
        if sys.version_info < (3, 10):
            tb = traceback.format_exception(error.__class__, error, error.__traceback__)
        else:
            tb = traceback.format_exception(error)
        return (msg + '\n\n'.join(tb)).strip()
    except:
        return msg