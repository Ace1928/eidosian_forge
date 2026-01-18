import inspect
import os
import platform
import socket
import urllib.parse as urllib_parse
import warnings
from collections.abc import Sequence
from functools import reduce
from html import escape
from http import cookiejar as cookielib
from io import IOBase, StringIO as NativeStringIO, TextIOBase
from sys import intern
from types import FrameType, MethodType as _MethodType
from typing import Any, AnyStr, cast
from urllib.parse import quote as urlquote, unquote as urlunquote
from incremental import Version
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
def _get_async_param(isAsync=None, **kwargs):
    """
    Provide a backwards-compatible way to get async param value that does not
    cause a syntax error under Python 3.7.

    @param isAsync: isAsync param value (should default to None)
    @type isAsync: L{bool}

    @param kwargs: keyword arguments of the caller (only async is allowed)
    @type kwargs: L{dict}

    @raise TypeError: Both isAsync and async specified.

    @return: Final isAsync param value
    @rtype: L{bool}
    """
    if 'async' in kwargs:
        warnings.warn("'async' keyword argument is deprecated, please use isAsync", DeprecationWarning, stacklevel=2)
    if isAsync is None and 'async' in kwargs:
        isAsync = kwargs.pop('async')
    if kwargs:
        raise TypeError
    return bool(isAsync)