from __future__ import annotations
import errno
import itertools
import mimetypes
import os
import time
import warnings
from html import escape
from typing import Any, Callable, Dict, Sequence
from urllib.parse import quote, unquote
from zope.interface import implementer
from incremental import Version
from typing_extensions import Literal
from twisted.internet import abstract, interfaces
from twisted.python import components, filepath, log
from twisted.python.compat import nativeString, networkString
from twisted.python.deprecate import deprecated
from twisted.python.runtime import platformType
from twisted.python.url import URL
from twisted.python.util import InsensitiveDict
from twisted.web import http, resource, server
from twisted.web.util import redirectTo
def _parseRangeHeader(self, range):
    """
        Parse the value of a Range header into (start, stop) pairs.

        In a given pair, either of start or stop can be None, signifying that
        no value was provided, but not both.

        @return: A list C{[(start, stop)]} of pairs of length at least one.

        @raise ValueError: if the header is syntactically invalid or if the
            Bytes-Unit is anything other than "bytes'.
        """
    try:
        kind, value = range.split(b'=', 1)
    except ValueError:
        raise ValueError("Missing '=' separator")
    kind = kind.strip()
    if kind != b'bytes':
        raise ValueError(f'Unsupported Bytes-Unit: {kind!r}')
    unparsedRanges = list(filter(None, map(bytes.strip, value.split(b','))))
    parsedRanges = []
    for byteRange in unparsedRanges:
        try:
            start, end = byteRange.split(b'-', 1)
        except ValueError:
            raise ValueError(f'Invalid Byte-Range: {byteRange!r}')
        if start:
            try:
                start = int(start)
            except ValueError:
                raise ValueError(f'Invalid Byte-Range: {byteRange!r}')
        else:
            start = None
        if end:
            try:
                end = int(end)
            except ValueError:
                raise ValueError(f'Invalid Byte-Range: {byteRange!r}')
        else:
            end = None
        if start is not None:
            if end is not None and start > end:
                raise ValueError(f'Invalid Byte-Range: {byteRange!r}')
        elif end is None:
            raise ValueError(f'Invalid Byte-Range: {byteRange!r}')
        parsedRanges.append((start, end))
    return parsedRanges