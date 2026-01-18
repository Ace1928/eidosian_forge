import threading
from types import TracebackType
from typing import Optional, Type
from ._exceptions import ExceptionMapping, PoolTimeout, map_exceptions
class ShieldCancellation:

    def __enter__(self) -> 'ShieldCancellation':
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]]=None, exc_value: Optional[BaseException]=None, traceback: Optional[TracebackType]=None) -> None:
        pass