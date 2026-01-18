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
def _deprecateAttribute(proxy, name, version, message):
    """
    Mark a module-level attribute as being deprecated.

    @type proxy: L{_ModuleProxy}
    @param proxy: The module proxy instance proxying the deprecated attributes

    @type name: C{str}
    @param name: Attribute name

    @type version: L{incremental.Version}
    @param version: Version that the attribute was deprecated in

    @type message: C{str}
    @param message: Deprecation message
    """
    _module = object.__getattribute__(proxy, '_module')
    attr = _DeprecatedAttribute(_module, name, version, message)
    _deprecatedAttributes = object.__getattribute__(proxy, '_deprecatedAttributes')
    _deprecatedAttributes[name] = attr