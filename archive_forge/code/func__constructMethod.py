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
def _constructMethod(cls, name, self):
    """
    Construct a bound method.

    @param cls: The class that the method should be bound to.
    @type cls: L{type}

    @param name: The name of the method.
    @type name: native L{str}

    @param self: The object that the method is bound to.
    @type self: any object

    @return: a bound method
    @rtype: L{_MethodType}
    """
    func = cls.__dict__[name]
    return _MethodType(func, self)