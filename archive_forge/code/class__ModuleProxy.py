from __future__ import annotations
import inspect
import sys
from dis import findlinestarts
from functools import wraps
from types import ModuleType
from typing import Any, Callable, Dict, Optional, TypeVar, cast
from warnings import warn, warn_explicit
from incremental import Version, getVersionString
from typing_extensions import ParamSpec
class _ModuleProxy:
    """
    Python module wrapper to hook module-level attribute access.

    Access to deprecated attributes first checks
    L{_ModuleProxy._deprecatedAttributes}, if the attribute does not appear
    there then access falls through to L{_ModuleProxy._module}, the wrapped
    module object.

    @ivar _module: Module on which to hook attribute access.
    @type _module: C{module}

    @ivar _deprecatedAttributes: Mapping of attribute names to objects that
        retrieve the module attribute's original value.
    @type _deprecatedAttributes: C{dict} mapping C{str} to
        L{_DeprecatedAttribute}

    @ivar _lastWasPath: Heuristic guess as to whether warnings about this
        package should be ignored for the next call.  If the last attribute
        access of this module was a C{getattr} of C{__path__}, we will assume
        that it was the import system doing it and we won't emit a warning for
        the next access, even if it is to a deprecated attribute.  The CPython
        import system always tries to access C{__path__}, then the attribute
        itself, then the attribute itself again, in both successful and failed
        cases.
    @type _lastWasPath: C{bool}
    """

    def __init__(self, module):
        state = _InternalState(self)
        state._module = module
        state._deprecatedAttributes = {}
        state._lastWasPath = False

    def __repr__(self) -> str:
        """
        Get a string containing the type of the module proxy and a
        representation of the wrapped module object.
        """
        state = _InternalState(self)
        return f'<{type(self).__name__} module={state._module!r}>'

    def __setattr__(self, name, value):
        """
        Set an attribute on the wrapped module object.
        """
        state = _InternalState(self)
        state._lastWasPath = False
        setattr(state._module, name, value)

    def __getattribute__(self, name):
        """
        Get an attribute from the module object, possibly emitting a warning.

        If the specified name has been deprecated, then a warning is issued.
        (Unless certain obscure conditions are met; see
        L{_ModuleProxy._lastWasPath} for more information about what might quash
        such a warning.)
        """
        state = _InternalState(self)
        if state._lastWasPath:
            deprecatedAttribute = None
        else:
            deprecatedAttribute = state._deprecatedAttributes.get(name)
        if deprecatedAttribute is not None:
            value = deprecatedAttribute.get()
        else:
            value = getattr(state._module, name)
        if name == '__path__':
            state._lastWasPath = True
        else:
            state._lastWasPath = False
        return value