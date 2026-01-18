import abc
import collections
import contextlib
import functools
import inspect
from concurrent.futures import CancelledError
from typing import (
import duet.impl as impl
from duet.aitertools import aenumerate, aiter, AnyIterable, AsyncCollector
from duet.futuretools import AwaitableFuture
class LimitedScope(abc.ABC):
    """Combined Scope (for running async iters) and Limiter (for throttling).

    Provides convenience methods for running coroutines in parallel within this
    scope while throttling to prevent iterators from running too far ahead.
    """

    @property
    @abc.abstractmethod
    def scope(self) -> Scope:
        pass

    @property
    @abc.abstractmethod
    def limiter(self) -> Limiter:
        pass

    def spawn(self, func: Callable[..., Awaitable[Any]], *args, **kwds) -> None:
        """Starts a background task that will run the given function."""
        self.scope.spawn(func, *args, **kwds)

    async def pmap_async(self, func: Callable[[T], Awaitable[U]], iterable: AnyIterable[T]) -> List[U]:
        return [x async for x in self.pmap_aiter(func, iterable)]

    def pmap_aiter(self, func: Callable[[T], Awaitable[U]], iterable: AnyIterable[T]) -> AsyncIterator[U]:
        return pmap_aiter(self.scope, func, self.limiter.throttle(iterable))

    async def pstarmap_async(self, func: Callable[..., Awaitable[U]], iterable: AnyIterable[Any]) -> List[U]:
        return [x async for x in self.pstarmap_aiter(func, iterable)]

    def pstarmap_aiter(self, func: Callable[..., Awaitable[U]], iterable: AnyIterable[Any]) -> AsyncIterator[U]:
        return pstarmap_aiter(self.scope, func, self.limiter.throttle(iterable))