from abc import ABCMeta, abstractmethod
import sys
class AsyncIterator(AsyncIterable):
    __slots__ = ()

    @abstractmethod
    async def __anext__(self):
        """Return the next item or raise StopAsyncIteration when exhausted."""
        raise StopAsyncIteration

    def __aiter__(self):
        return self

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AsyncIterator:
            return _check_methods(C, '__anext__', '__aiter__')
        return NotImplemented