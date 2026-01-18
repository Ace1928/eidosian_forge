import typing
import warnings
from ..api import BytesBackend
from ..api import NO_VALUE
class _RedisLockWrapper:
    __slots__ = ('mutex', '__weakref__')

    def __init__(self, mutex: typing.Any):
        self.mutex = mutex

    def acquire(self, wait: bool=True) -> typing.Any:
        return self.mutex.acquire(blocking=wait)

    def release(self) -> typing.Any:
        return self.mutex.release()

    def locked(self) -> bool:
        return self.mutex.locked()