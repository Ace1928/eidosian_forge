from __future__ import annotations
from typing import (
import warnings
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
class DirNamesMixin:
    _accessors: set[str] = set()
    _hidden_attrs: frozenset[str] = frozenset()

    @final
    def _dir_deletions(self) -> set[str]:
        """
        Delete unwanted __dir__ for this object.
        """
        return self._accessors | self._hidden_attrs

    def _dir_additions(self) -> set[str]:
        """
        Add additional __dir__ for this object.
        """
        return {accessor for accessor in self._accessors if hasattr(self, accessor)}

    def __dir__(self) -> list[str]:
        """
        Provide method name lookup and completion.

        Notes
        -----
        Only provide 'public' methods.
        """
        rv = set(super().__dir__())
        rv = rv - self._dir_deletions() | self._dir_additions()
        return sorted(rv)