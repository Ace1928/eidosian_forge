import threading
from concurrent.futures import Future
from typing import Any, Callable, Generator, Generic, Optional, Tuple, Type, TypeVar
class AwaitableFuture(Future, Generic[T]):
    """A Future that can be awaited."""
    _condition: threading.Condition

    @staticmethod
    def isfuture(value: Any) -> bool:
        return isinstance(value, FutureClasses)

    @staticmethod
    def wrap(future: FutureLike[T]) -> 'AwaitableFuture[T]':
        """Creates an awaitable future that wraps the given source future."""
        awaitable = AwaitableFuture[T]()

        def cancel(awaitable_future: Future):
            if awaitable_future.cancelled():
                future.cancel()

        def callback(future: FutureLike[T]):
            if future.cancelled():
                awaitable.cancel()
            else:
                error = future.exception()
                if error is None:
                    awaitable.try_set_result(future.result())
                else:
                    awaitable.try_set_exception(error)
        awaitable.add_done_callback(cancel)
        future.add_done_callback(callback)
        return awaitable

    def __await__(self) -> Generator['AwaitableFuture[T]', None, T]:
        yield self
        return self.result()

    def try_set_result(self, result: T) -> bool:
        """Sets the result on this future if not already done.

        Returns:
            True if we set the result, False if the future was already done.
        """
        with self._condition:
            if self.done():
                return False
            self.set_result(result)
            return True

    def try_set_exception(self, exception: Optional[BaseException]) -> bool:
        """Sets an exception on this future if not already done.

        Returns:
            True if we set the exception, False if the future was already done.
        """
        with self._condition:
            if self.done():
                return False
            self.set_exception(exception)
            return True