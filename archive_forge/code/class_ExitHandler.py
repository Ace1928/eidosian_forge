from __future__ import annotations
import collections.abc as c
import contextlib
import json
import os
import re
import shlex
import sys
import tempfile
import textwrap
import typing as t
from .constants import (
from .encoding import (
from .util import (
from .io import (
from .data import (
from .provider.layout import (
from .host_configs import (
class ExitHandler:
    """Simple exit handler implementation."""
    _callbacks: list[tuple[t.Callable, tuple[t.Any, ...], dict[str, t.Any]]] = []

    @staticmethod
    def register(func: t.Callable, *args, **kwargs) -> None:
        """Register the given function and args as a callback to execute during program termination."""
        ExitHandler._callbacks.append((func, args, kwargs))

    @staticmethod
    @contextlib.contextmanager
    def context() -> t.Generator[None, None, None]:
        """Run all registered handlers when the context is exited."""
        last_exception: BaseException | None = None
        try:
            yield
        finally:
            queue = list(ExitHandler._callbacks)
            while queue:
                func, args, kwargs = queue.pop()
                try:
                    func(*args, **kwargs)
                except BaseException as ex:
                    last_exception = ex
                    display.fatal(f'Exit handler failed: {ex}')
            if last_exception:
                raise last_exception