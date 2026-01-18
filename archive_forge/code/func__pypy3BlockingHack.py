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
def _pypy3BlockingHack():
    """
    Work around U{https://foss.heptapod.net/pypy/pypy/-/issues/3051}
    by replacing C{socket.fromfd} with a more conservative version.
    """
    try:
        from fcntl import F_GETFL, F_SETFL, fcntl
    except ImportError:
        return
    if not _PYPY:
        return

    def fromFDWithoutModifyingFlags(fd, family, type, proto=None):
        passproto = [proto] * (proto is not None)
        flags = fcntl(fd, F_GETFL)
        try:
            return realFromFD(fd, family, type, *passproto)
        finally:
            fcntl(fd, F_SETFL, flags)
    realFromFD = socket.fromfd
    if realFromFD.__name__ == fromFDWithoutModifyingFlags.__name__:
        return
    socket.fromfd = fromFDWithoutModifyingFlags