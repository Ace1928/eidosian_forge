from __future__ import annotations
import asyncio
import inspect
from asyncio import InvalidStateError, Task
from enum import Enum
from typing import TYPE_CHECKING, Awaitable, Optional, Union
class SetupMode(Enum):
    """Setup mode for AstraDBEnvironment as enumerator."""
    SYNC = 1
    ASYNC = 2
    OFF = 3