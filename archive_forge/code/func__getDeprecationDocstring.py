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
def _getDeprecationDocstring(version, replacement=None):
    """
    Generate an addition to a deprecated object's docstring that explains its
    deprecation.

    @param version: the version it was deprecated.
    @type version: L{incremental.Version}

    @param replacement: The replacement, if specified.
    @type replacement: C{str} or callable

    @return: a string like "Deprecated in Twisted 27.2.0; please use
        twisted.timestream.tachyon.flux instead."
    """
    doc = f'Deprecated in {getVersionString(version)}'
    if replacement:
        doc = f'{doc}; {_getReplacementString(replacement)}'
    return doc + '.'