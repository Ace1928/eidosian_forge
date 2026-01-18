from __future__ import annotations
import asyncio  # noqa
import typing
from typing import Any
from typing import Callable
from typing import Coroutine
from typing import TypeVar
Run sync function in greenlet. Support nested calls