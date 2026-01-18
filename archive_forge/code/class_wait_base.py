import abc
import random
import typing
from pip._vendor.tenacity import _utils
class wait_base(abc.ABC):
    """Abstract base class for wait strategies."""

    @abc.abstractmethod
    def __call__(self, retry_state: 'RetryCallState') -> float:
        pass

    def __add__(self, other: 'wait_base') -> 'wait_combine':
        return wait_combine(self, other)

    def __radd__(self, other: 'wait_base') -> typing.Union['wait_combine', 'wait_base']:
        if other == 0:
            return self
        return self.__add__(other)