import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
import numpy
from .compat import cupy, has_cupy
@dataclass
class ArgsKwargs:
    """A tuple of (args, kwargs) that can be spread into some function f:

    f(*args, **kwargs)
    """
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

    @classmethod
    def from_items(cls, items: Sequence[Tuple[Union[int, str], Any]]) -> 'ArgsKwargs':
        """Create an ArgsKwargs object from a sequence of (key, value) tuples,
        such as produced by argskwargs.items(). Each key should be either a string
        or an integer. Items with int keys are added to the args list, and
        items with string keys are added to the kwargs list. The args list is
        determined by sequence order, not the value of the integer.
        """
        args = []
        kwargs = {}
        for key, value in items:
            if isinstance(key, int):
                args.append(value)
            else:
                kwargs[key] = value
        return cls(args=tuple(args), kwargs=kwargs)

    def keys(self) -> Iterable[Union[int, str]]:
        """Yield indices from self.args, followed by keys from self.kwargs."""
        yield from range(len(self.args))
        yield from self.kwargs.keys()

    def values(self) -> Iterable[Any]:
        """Yield elements of from self.args, followed by values from self.kwargs."""
        yield from self.args
        yield from self.kwargs.values()

    def items(self) -> Iterable[Tuple[Union[int, str], Any]]:
        """Yield enumerate(self.args), followed by self.kwargs.items()"""
        yield from enumerate(self.args)
        yield from self.kwargs.items()