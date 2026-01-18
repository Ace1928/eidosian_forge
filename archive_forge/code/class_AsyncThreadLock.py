import threading
from types import TracebackType
from typing import Optional, Type
from ._exceptions import ExceptionMapping, PoolTimeout, map_exceptions
class AsyncThreadLock:
    """
    This is a threading-only lock for no-I/O contexts.

    In the sync case `ThreadLock` provides thread locking.
    In the async case `AsyncThreadLock` is a no-op.
    """

    def __enter__(self) -> 'AsyncThreadLock':
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]]=None, exc_value: Optional[BaseException]=None, traceback: Optional[TracebackType]=None) -> None:
        pass