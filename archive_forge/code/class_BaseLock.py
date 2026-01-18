from typing import Optional, Type, IO
from ._typing_compat import Literal
from types import TracebackType
class BaseLock:
    """Base class for file locking"""

    def __init__(self) -> None:
        self.locked = False

    def acquire(self) -> None:
        pass

    def release(self) -> None:
        pass

    def __enter__(self) -> 'BaseLock':
        self.acquire()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], exc_tb: Optional[TracebackType]) -> Literal[False]:
        if self.locked:
            self.release()
        return False

    def __del__(self) -> None:
        if self.locked:
            self.release()