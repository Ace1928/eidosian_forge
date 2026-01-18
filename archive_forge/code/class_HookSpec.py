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
@final
class HookSpec:
    __slots__ = ('namespace', 'function', 'name', 'argnames', 'kwargnames', 'opts', 'warn_on_impl')

    def __init__(self, namespace: _Namespace, name: str, opts: HookspecOpts) -> None:
        self.namespace = namespace
        self.function: Callable[..., object] = getattr(namespace, name)
        self.name = name
        self.argnames, self.kwargnames = varnames(self.function)
        self.opts = opts
        self.warn_on_impl = opts.get('warn_on_impl')