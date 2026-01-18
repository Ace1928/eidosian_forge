import sys
import asyncio
import functools
from typing import Optional, Any, Callable, Awaitable, Union, TypeVar, Coroutine, Iterable, AsyncIterable, AsyncIterator, AsyncGenerator
from lazyops.utils.system import get_cpu_count
from lazyops.utils.pooler import ThreadPooler
from lazyops.types.common import UpperStrEnum

        Get the value of the return when type
        