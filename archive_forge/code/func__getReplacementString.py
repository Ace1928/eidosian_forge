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
def _getReplacementString(replacement):
    """
    Surround a replacement for a deprecated API with some polite text exhorting
    the user to consider it as an alternative.

    @type replacement: C{str} or callable

    @return: a string like "please use twisted.python.modules.getModule
        instead".
    """
    if callable(replacement):
        replacement = _fullyQualifiedName(replacement)
    return f'please use {replacement} instead'