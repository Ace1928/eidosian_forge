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
class ClassProvider(Provider, Generic[T]):
    """Provides instances from a given class, created using an Injector."""

    def __init__(self, cls: Type[T]) -> None:
        self._cls = cls

    def get(self, injector: 'Injector') -> T:
        return injector.create_object(self._cls)