import asyncio
import enum
import sys
import warnings
from types import TracebackType
from typing import Optional, Type
def _uncancel_task(task: 'asyncio.Task[object]') -> None:
    pass