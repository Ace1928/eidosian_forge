from __future__ import annotations
import inspect
import types
import warnings
from typing import Any
from typing import Callable
from typing import cast
from typing import Final
from typing import Iterable
from typing import Mapping
from typing import Sequence
from typing import TYPE_CHECKING
from . import _tracing
from ._callers import _multicall
from ._hooks import _HookImplFunction
from ._hooks import _Namespace
from ._hooks import _Plugin
from ._hooks import _SubsetHookCaller
from ._hooks import HookCaller
from ._hooks import HookImpl
from ._hooks import HookimplOpts
from ._hooks import HookRelay
from ._hooks import HookspecOpts
from ._hooks import normalize_hookimpl_opts
from ._result import Result
def _verify_hook(self, hook: HookCaller, hookimpl: HookImpl) -> None:
    if hook.is_historic() and (hookimpl.hookwrapper or hookimpl.wrapper):
        raise PluginValidationError(hookimpl.plugin, 'Plugin %r\nhook %r\nhistoric incompatible with yield/wrapper/hookwrapper' % (hookimpl.plugin_name, hook.name))
    assert hook.spec is not None
    if hook.spec.warn_on_impl:
        _warn_for_function(hook.spec.warn_on_impl, hookimpl.function)
    notinspec = set(hookimpl.argnames) - set(hook.spec.argnames)
    if notinspec:
        raise PluginValidationError(hookimpl.plugin, 'Plugin %r for hook %r\nhookimpl definition: %s\nArgument(s) %s are declared in the hookimpl but can not be found in the hookspec' % (hookimpl.plugin_name, hook.name, _formatdef(hookimpl.function), notinspec))
    if (hookimpl.wrapper or hookimpl.hookwrapper) and (not inspect.isgeneratorfunction(hookimpl.function)):
        raise PluginValidationError(hookimpl.plugin, 'Plugin %r for hook %r\nhookimpl definition: %s\nDeclared as wrapper=True or hookwrapper=True but function is not a generator function' % (hookimpl.plugin_name, hook.name, _formatdef(hookimpl.function)))
    if hookimpl.wrapper and hookimpl.hookwrapper:
        raise PluginValidationError(hookimpl.plugin, 'Plugin %r for hook %r\nhookimpl definition: %s\nThe wrapper=True and hookwrapper=True options are mutually exclusive' % (hookimpl.plugin_name, hook.name, _formatdef(hookimpl.function)))