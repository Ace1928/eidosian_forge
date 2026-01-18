from __future__ import annotations
from abc import ABCMeta, abstractmethod, abstractproperty
from contextlib import contextmanager
from typing import Callable, ContextManager, Generator
from prompt_toolkit.key_binding import KeyPress
class DummyInput(Input):
    """
    Input for use in a `DummyApplication`

    If used in an actual application, it will make the application render
    itself once and exit immediately, due to an `EOFError`.
    """

    def fileno(self) -> int:
        raise NotImplementedError

    def typeahead_hash(self) -> str:
        return 'dummy-%s' % id(self)

    def read_keys(self) -> list[KeyPress]:
        return []

    @property
    def closed(self) -> bool:
        return True

    def raw_mode(self) -> ContextManager[None]:
        return _dummy_context_manager()

    def cooked_mode(self) -> ContextManager[None]:
        return _dummy_context_manager()

    def attach(self, input_ready_callback: Callable[[], None]) -> ContextManager[None]:
        input_ready_callback()
        return _dummy_context_manager()

    def detach(self) -> ContextManager[None]:
        return _dummy_context_manager()