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
class HookspecMarker:
    """Decorator for marking functions as hook specifications.

    Instantiate it with a project_name to get a decorator.
    Calling :meth:`PluginManager.add_hookspecs` later will discover all marked
    functions if the :class:`PluginManager` uses the same project name.
    """
    __slots__ = ('project_name',)

    def __init__(self, project_name: str) -> None:
        self.project_name: Final = project_name

    @overload
    def __call__(self, function: _F, firstresult: bool=False, historic: bool=False, warn_on_impl: Warning | None=None) -> _F:
        ...

    @overload
    def __call__(self, function: None=..., firstresult: bool=..., historic: bool=..., warn_on_impl: Warning | None=...) -> Callable[[_F], _F]:
        ...

    def __call__(self, function: _F | None=None, firstresult: bool=False, historic: bool=False, warn_on_impl: Warning | None=None) -> _F | Callable[[_F], _F]:
        """If passed a function, directly sets attributes on the function
        which will make it discoverable to :meth:`PluginManager.add_hookspecs`.

        If passed no function, returns a decorator which can be applied to a
        function later using the attributes supplied.

        :param firstresult:
            If ``True``, the 1:N hook call (N being the number of registered
            hook implementation functions) will stop at I<=N when the I'th
            function returns a non-``None`` result. See :ref:`firstresult`.

        :param historic:
            If ``True``, every call to the hook will be memorized and replayed
            on plugins registered after the call was made. See :ref:`historic`.

        :param warn_on_impl:
            If given, every implementation of this hook will trigger the given
            warning. See :ref:`warn_on_impl`.
        """

        def setattr_hookspec_opts(func: _F) -> _F:
            if historic and firstresult:
                raise ValueError('cannot have a historic firstresult hook')
            opts: HookspecOpts = {'firstresult': firstresult, 'historic': historic, 'warn_on_impl': warn_on_impl}
            setattr(func, self.project_name + '_spec', opts)
            return func
        if function is not None:
            return setattr_hookspec_opts(function)
        else:
            return setattr_hookspec_opts