import atexit
import builtins
import functools
import inspect
import os
import operator
import timeit
import math
import sys
import traceback
import weakref
import warnings
import threading
import contextlib
import typing as _tp
from types import ModuleType
from importlib import import_module
import numpy as np
from inspect import signature as pysignature # noqa: F401
from inspect import Signature as pySignature # noqa: F401
from inspect import Parameter as pyParameter # noqa: F401
from numba.core.config import (PYVERSION, MACHINE_BITS, # noqa: F401
from numba.core import config
from numba.core import types
from collections.abc import Mapping, Sequence, MutableSet, MutableMapping
class ThreadLocalStack:
    """A TLS stack container.

    Uses the BORG pattern and stores states in threadlocal storage.
    """
    _tls = threading.local()
    stack_name: str
    _registered = {}

    def __init_subclass__(cls, *, stack_name, **kwargs):
        super().__init_subclass__(**kwargs)
        assert stack_name not in cls._registered, f"stack_name: '{stack_name}' already in use"
        cls.stack_name = stack_name
        cls._registered[stack_name] = cls

    def __init__(self):
        assert type(self) is not ThreadLocalStack
        tls = self._tls
        attr = f'stack_{self.stack_name}'
        try:
            tls_stack = getattr(tls, attr)
        except AttributeError:
            tls_stack = list()
            setattr(tls, attr, tls_stack)
        self._stack = tls_stack

    def push(self, state):
        """Push to the stack
        """
        self._stack.append(state)

    def pop(self):
        """Pop from the stack
        """
        return self._stack.pop()

    def top(self):
        """Get the top item on the stack.

        Raises IndexError if the stack is empty. Users should check the size
        of the stack beforehand.
        """
        return self._stack[-1]

    def __len__(self):
        return len(self._stack)

    @contextlib.contextmanager
    def enter(self, state):
        """A contextmanager that pushes ``state`` for the duration of the
        context.
        """
        self.push(state)
        try:
            yield
        finally:
            self.pop()