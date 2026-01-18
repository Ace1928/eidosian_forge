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
def _getDeprecationWarningString(fqpn, version, format=None, replacement=None):
    """
    Return a string indicating that the Python name was deprecated in the given
    version.

    @param fqpn: Fully qualified Python name of the thing being deprecated
    @type fqpn: C{str}

    @param version: Version that C{fqpn} was deprecated in.
    @type version: L{incremental.Version}

    @param format: A user-provided format to interpolate warning values into, or
        L{DEPRECATION_WARNING_FORMAT
        <twisted.python.deprecate.DEPRECATION_WARNING_FORMAT>} if L{None} is
        given.
    @type format: C{str}

    @param replacement: what should be used in place of C{fqpn}. Either pass in
        a string, which will be inserted into the warning message, or a
        callable, which will be expanded to its full import path.
    @type replacement: C{str} or callable

    @return: A textual description of the deprecation
    @rtype: C{str}
    """
    if format is None:
        format = DEPRECATION_WARNING_FORMAT
    warningString = format % {'fqpn': fqpn, 'version': getVersionString(version)}
    if replacement:
        warningString = '{}; {}'.format(warningString, _getReplacementString(replacement))
    return warningString