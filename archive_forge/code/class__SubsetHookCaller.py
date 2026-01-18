from __future__ import annotations
import inspect
import sys
import warnings
from types import ModuleType
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import Final
from typing import final
from typing import Generator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypedDict
from typing import TypeVar
from typing import Union
from ._result import Result
class _SubsetHookCaller(HookCaller):
    """A proxy to another HookCaller which manages calls to all registered
    plugins except the ones from remove_plugins."""
    __slots__ = ('_orig', '_remove_plugins')

    def __init__(self, orig: HookCaller, remove_plugins: AbstractSet[_Plugin]) -> None:
        self._orig = orig
        self._remove_plugins = remove_plugins
        self.name = orig.name
        self._hookexec = orig._hookexec

    @property
    def _hookimpls(self) -> list[HookImpl]:
        return [impl for impl in self._orig._hookimpls if impl.plugin not in self._remove_plugins]

    @property
    def spec(self) -> HookSpec | None:
        return self._orig.spec

    @property
    def _call_history(self) -> _CallHistory | None:
        return self._orig._call_history

    def __repr__(self) -> str:
        return f'<_SubsetHookCaller {self.name!r}>'