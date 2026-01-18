from __future__ import annotations
from abc import ABCMeta, abstractmethod, abstractproperty
from inspect import isawaitable
from typing import (
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.filters import FilterOrBool, Never, to_filter
from prompt_toolkit.keys import KEY_ALIASES, Keys
class _Proxy(KeyBindingsBase):
    """
    Common part for ConditionalKeyBindings and _MergedKeyBindings.
    """

    def __init__(self) -> None:
        self._bindings2: KeyBindingsBase = KeyBindings()
        self._last_version: Hashable = ()

    def _update_cache(self) -> None:
        """
        If `self._last_version` is outdated, then this should update
        the version and `self._bindings2`.
        """
        raise NotImplementedError

    @property
    def bindings(self) -> list[Binding]:
        self._update_cache()
        return self._bindings2.bindings

    @property
    def _version(self) -> Hashable:
        self._update_cache()
        return self._last_version

    def get_bindings_for_keys(self, keys: KeysTuple) -> list[Binding]:
        self._update_cache()
        return self._bindings2.get_bindings_for_keys(keys)

    def get_bindings_starting_with_keys(self, keys: KeysTuple) -> list[Binding]:
        self._update_cache()
        return self._bindings2.get_bindings_starting_with_keys(keys)