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
class HookCaller:
    """A caller of all registered implementations of a hook specification."""
    __slots__ = ('name', 'spec', '_hookexec', '_hookimpls', '_call_history')

    def __init__(self, name: str, hook_execute: _HookExec, specmodule_or_class: _Namespace | None=None, spec_opts: HookspecOpts | None=None) -> None:
        """:meta private:"""
        self.name: Final = name
        self._hookexec: Final = hook_execute
        self._hookimpls: Final[list[HookImpl]] = []
        self._call_history: _CallHistory | None = None
        self.spec: HookSpec | None = None
        if specmodule_or_class is not None:
            assert spec_opts is not None
            self.set_specification(specmodule_or_class, spec_opts)

    def has_spec(self) -> bool:
        return self.spec is not None

    def set_specification(self, specmodule_or_class: _Namespace, spec_opts: HookspecOpts) -> None:
        if self.spec is not None:
            raise ValueError(f'Hook {self.spec.name!r} is already registered within namespace {self.spec.namespace}')
        self.spec = HookSpec(specmodule_or_class, self.name, spec_opts)
        if spec_opts.get('historic'):
            self._call_history = []

    def is_historic(self) -> bool:
        """Whether this caller is :ref:`historic <historic>`."""
        return self._call_history is not None

    def _remove_plugin(self, plugin: _Plugin) -> None:
        for i, method in enumerate(self._hookimpls):
            if method.plugin == plugin:
                del self._hookimpls[i]
                return
        raise ValueError(f'plugin {plugin!r} not found')

    def get_hookimpls(self) -> list[HookImpl]:
        """Get all registered hook implementations for this hook."""
        return self._hookimpls.copy()

    def _add_hookimpl(self, hookimpl: HookImpl) -> None:
        """Add an implementation to the callback chain."""
        for i, method in enumerate(self._hookimpls):
            if method.hookwrapper or method.wrapper:
                splitpoint = i
                break
        else:
            splitpoint = len(self._hookimpls)
        if hookimpl.hookwrapper or hookimpl.wrapper:
            start, end = (splitpoint, len(self._hookimpls))
        else:
            start, end = (0, splitpoint)
        if hookimpl.trylast:
            self._hookimpls.insert(start, hookimpl)
        elif hookimpl.tryfirst:
            self._hookimpls.insert(end, hookimpl)
        else:
            i = end - 1
            while i >= start and self._hookimpls[i].tryfirst:
                i -= 1
            self._hookimpls.insert(i + 1, hookimpl)

    def __repr__(self) -> str:
        return f'<HookCaller {self.name!r}>'

    def _verify_all_args_are_provided(self, kwargs: Mapping[str, object]) -> None:
        if self.spec:
            for argname in self.spec.argnames:
                if argname not in kwargs:
                    notincall = ', '.join((repr(argname) for argname in self.spec.argnames if argname not in kwargs.keys()))
                    warnings.warn('Argument(s) {} which are declared in the hookspec cannot be found in this hook call'.format(notincall), stacklevel=2)
                    break

    def __call__(self, **kwargs: object) -> Any:
        """Call the hook.

        Only accepts keyword arguments, which should match the hook
        specification.

        Returns the result(s) of calling all registered plugins, see
        :ref:`calling`.
        """
        assert not self.is_historic(), 'Cannot directly call a historic hook - use call_historic instead.'
        self._verify_all_args_are_provided(kwargs)
        firstresult = self.spec.opts.get('firstresult', False) if self.spec else False
        return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)

    def call_historic(self, result_callback: Callable[[Any], None] | None=None, kwargs: Mapping[str, object] | None=None) -> None:
        """Call the hook with given ``kwargs`` for all registered plugins and
        for all plugins which will be registered afterwards, see
        :ref:`historic`.

        :param result_callback:
            If provided, will be called for each non-``None`` result obtained
            from a hook implementation.
        """
        assert self._call_history is not None
        kwargs = kwargs or {}
        self._verify_all_args_are_provided(kwargs)
        self._call_history.append((kwargs, result_callback))
        res = self._hookexec(self.name, self._hookimpls.copy(), kwargs, False)
        if result_callback is None:
            return
        if isinstance(res, list):
            for x in res:
                result_callback(x)

    def call_extra(self, methods: Sequence[Callable[..., object]], kwargs: Mapping[str, object]) -> Any:
        """Call the hook with some additional temporarily participating
        methods using the specified ``kwargs`` as call parameters, see
        :ref:`call_extra`."""
        assert not self.is_historic(), 'Cannot directly call a historic hook - use call_historic instead.'
        self._verify_all_args_are_provided(kwargs)
        opts: HookimplOpts = {'wrapper': False, 'hookwrapper': False, 'optionalhook': False, 'trylast': False, 'tryfirst': False, 'specname': None}
        hookimpls = self._hookimpls.copy()
        for method in methods:
            hookimpl = HookImpl(None, '<temp>', method, opts)
            i = len(hookimpls) - 1
            while i >= 0 and ((hookimpls[i].hookwrapper or hookimpls[i].wrapper) or hookimpls[i].tryfirst):
                i -= 1
            hookimpls.insert(i + 1, hookimpl)
        firstresult = self.spec.opts.get('firstresult', False) if self.spec else False
        return self._hookexec(self.name, hookimpls, kwargs, firstresult)

    def _maybe_apply_history(self, method: HookImpl) -> None:
        """Apply call history to a new hookimpl if it is marked as historic."""
        if self.is_historic():
            assert self._call_history is not None
            for kwargs, result_callback in self._call_history:
                res = self._hookexec(self.name, [method], kwargs, False)
                if res and result_callback is not None:
                    assert isinstance(res, list)
                    result_callback(res[0])