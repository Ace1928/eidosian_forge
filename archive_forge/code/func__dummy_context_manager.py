from __future__ import annotations
from abc import ABCMeta, abstractmethod, abstractproperty
from contextlib import contextmanager
from typing import Callable, ContextManager, Generator
from prompt_toolkit.key_binding import KeyPress
@contextmanager
def _dummy_context_manager() -> Generator[None, None, None]:
    yield