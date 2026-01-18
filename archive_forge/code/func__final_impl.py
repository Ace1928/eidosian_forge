from __future__ import annotations
import collections.abc
import inspect
import os
import signal
import threading
from abc import ABCMeta
from functools import update_wrapper
from typing import (
from sniffio import thread_local as sniffio_loop
import trio
def _final_impl(decorated: type[T]) -> type[T]:
    """Decorator that enforces a class to be final (i.e., subclass not allowed).

    If a class uses this metaclass like this::

        @final
        class SomeClass:
            pass

    The metaclass will ensure that no subclass can be created.

    Raises
    ------
    - TypeError if a subclass is created
    """
    decorated.__init_subclass__ = classmethod(_init_final_cls)
    return std_final(decorated)