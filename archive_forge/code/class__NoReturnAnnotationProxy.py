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
class _NoReturnAnnotationProxy:

    def __init__(self, callable: Callable) -> None:
        self.callable = callable

    def __getattribute__(self, name: str) -> Any:
        callable = object.__getattribute__(self, 'callable')
        if name == '__annotations__':
            annotations = callable.__annotations__
            return {name: value for name, value in annotations.items() if name != 'return'}
        return getattr(callable, name)