import asyncio
import base64
import binascii
import contextlib
import datetime
import enum
import functools
import inspect
import netrc
import os
import platform
import re
import sys
import time
import warnings
import weakref
from collections import namedtuple
from contextlib import suppress
from email.parser import HeaderParser
from email.utils import parsedate
from math import ceil
from pathlib import Path
from types import TracebackType
from typing import (
from urllib.parse import quote
from urllib.request import getproxies, proxy_bypass
import attr
from multidict import MultiDict, MultiDictProxy, MultiMapping
from yarl import URL
from . import hdrs
from .log import client_logger, internal_logger
@functools.total_ordering
class AppKey(Generic[_T]):
    """Keys for static typing support in Application."""
    __slots__ = ('_name', '_t', '__orig_class__')
    __orig_class__: Type[object]

    def __init__(self, name: str, t: Optional[Type[_T]]=None):
        frame = inspect.currentframe()
        while frame:
            if frame.f_code.co_name == '<module>':
                module: str = frame.f_globals['__name__']
                break
            frame = frame.f_back
        self._name = module + '.' + name
        self._t = t

    def __lt__(self, other: object) -> bool:
        if isinstance(other, AppKey):
            return self._name < other._name
        return True

    def __repr__(self) -> str:
        t = self._t
        if t is None:
            with suppress(AttributeError):
                t = get_args(self.__orig_class__)[0]
        if t is None:
            t_repr = '<<Unknown>>'
        elif isinstance(t, type):
            if t.__module__ == 'builtins':
                t_repr = t.__qualname__
            else:
                t_repr = f'{t.__module__}.{t.__qualname__}'
        else:
            t_repr = repr(t)
        return f'<AppKey({self._name}, type={t_repr})>'