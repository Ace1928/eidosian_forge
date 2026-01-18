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
class MultiBindProvider(ListOfProviders[List[T]]):
    """Used by :meth:`Binder.multibind` to flatten results of providers that
    return sequences."""

    def get(self, injector: 'Injector') -> List[T]:
        return [i for provider in self._providers for i in provider.get(injector)]