import collections.abc
import gc
import inspect
import re
import sys
import weakref
from functools import partial, wraps
from itertools import chain
from typing import (
from scrapy.utils.asyncgen import as_async_generator
class MutableChain(Iterable):
    """
    Thin wrapper around itertools.chain, allowing to add iterables "in-place"
    """

    def __init__(self, *args: Iterable):
        self.data = chain.from_iterable(args)

    def extend(self, *iterables: Iterable) -> None:
        self.data = chain(self.data, chain.from_iterable(iterables))

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> Any:
        return next(self.data)