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
class ThreadLocalScope(Scope):
    """A :class:`Scope` that returns a per-thread instance for a key."""

    def configure(self) -> None:
        self._locals = threading.local()

    def get(self, key: Type[T], provider: Provider[T]) -> Provider[T]:
        try:
            return getattr(self._locals, repr(key))
        except AttributeError:
            provider = InstanceProvider(provider.get(self.injector))
            setattr(self._locals, repr(key), provider)
            return provider