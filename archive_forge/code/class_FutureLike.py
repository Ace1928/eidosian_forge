import threading
from concurrent.futures import Future
from typing import Any, Callable, Generator, Generic, Optional, Tuple, Type, TypeVar
class FutureLike(Protocol[T]):

    def result(self) -> T:
        ...

    def exception(self) -> Optional[BaseException]:
        ...

    def add_done_callback(self, fn: Callable[['FutureLike[T]'], Any]) -> None:
        ...

    def cancel(self) -> bool:
        ...

    def cancelled(self) -> bool:
        ...