from __future__ import annotations
from abc import ABCMeta, abstractmethod, abstractproperty
from inspect import isawaitable
from typing import (
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.filters import FilterOrBool, Never, to_filter
from prompt_toolkit.keys import KEY_ALIASES, Keys
class _MergedKeyBindings(_Proxy):
    """
    Merge multiple registries of key bindings into one.

    This class acts as a proxy to multiple :class:`.KeyBindings` objects, but
    behaves as if this is just one bigger :class:`.KeyBindings`.

    :param registries: List of :class:`.KeyBindings` objects.
    """

    def __init__(self, registries: Sequence[KeyBindingsBase]) -> None:
        _Proxy.__init__(self)
        self.registries = registries

    def _update_cache(self) -> None:
        """
        If one of the original registries was changed. Update our merged
        version.
        """
        expected_version = tuple((r._version for r in self.registries))
        if self._last_version != expected_version:
            bindings2 = KeyBindings()
            for reg in self.registries:
                bindings2.bindings.extend(reg.bindings)
            self._bindings2 = bindings2
            self._last_version = expected_version