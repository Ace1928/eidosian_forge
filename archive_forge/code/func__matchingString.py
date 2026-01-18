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
def _matchingString(constantString, inputString):
    """
    Some functions, such as C{os.path.join}, operate on string arguments which
    may be bytes or text, and wish to return a value of the same type.  In
    those cases you may wish to have a string constant (in the case of
    C{os.path.join}, that constant would be C{os.path.sep}) involved in the
    parsing or processing, that must be of a matching type in order to use
    string operations on it.  L{_matchingString} will take a constant string
    (either L{bytes} or L{str}) and convert it to the same type as the
    input string.  C{constantString} should contain only characters from ASCII;
    to ensure this, it will be encoded or decoded regardless.

    @param constantString: A string literal used in processing.
    @type constantString: L{str} or L{bytes}

    @param inputString: A byte string or text string provided by the user.
    @type inputString: L{str} or L{bytes}

    @return: C{constantString} converted into the same type as C{inputString}
    @rtype: the type of C{inputString}
    """
    if isinstance(constantString, bytes):
        otherType = constantString.decode('ascii')
    else:
        otherType = constantString.encode('ascii')
    if type(constantString) == type(inputString):
        return constantString
    else:
        return otherType