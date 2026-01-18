import functools
import inspect
import itertools
import logging
import sys
import threading
import types
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import (
@private
class ListOfProviders(Provider, Generic[T]):
    """Provide a list of instances via other Providers."""
    _providers: List[Provider[T]]

    def __init__(self) -> None:
        self._providers = []

    def append(self, provider: Provider[T]) -> None:
        self._providers.append(provider)

    def __repr__(self) -> str:
        return '%s(%r)' % (type(self).__name__, self._providers)