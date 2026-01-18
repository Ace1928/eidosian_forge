from __future__ import annotations
from abc import ABCMeta, abstractmethod, abstractproperty
from inspect import isawaitable
from typing import (
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.filters import FilterOrBool, Never, to_filter
from prompt_toolkit.keys import KEY_ALIASES, Keys
class GlobalOnlyKeyBindings(_Proxy):
    """
    Wrapper around a :class:`.KeyBindings` object that only exposes the global
    key bindings.
    """

    def __init__(self, key_bindings: KeyBindingsBase) -> None:
        _Proxy.__init__(self)
        self.key_bindings = key_bindings

    def _update_cache(self) -> None:
        """
        If one of the original registries was changed. Update our merged
        version.
        """
        expected_version = self.key_bindings._version
        if self._last_version != expected_version:
            bindings2 = KeyBindings()
            for b in self.key_bindings.bindings:
                if b.is_global():
                    bindings2.bindings.append(b)
            self._bindings2 = bindings2
            self._last_version = expected_version