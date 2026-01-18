import collections.abc
import re
from typing import (
import warnings
from io import BytesIO
from datetime import datetime
from base64 import b64encode, b64decode
from numbers import Integral
from types import SimpleNamespace
from functools import singledispatch
from fontTools.misc import etree
from fontTools.misc.textTools import tostr
def _string_or_data_element(raw_bytes: bytes, ctx: SimpleNamespace) -> etree.Element:
    if ctx.use_builtin_types:
        return _data_element(raw_bytes, ctx)
    else:
        try:
            string = raw_bytes.decode(encoding='ascii', errors='strict')
        except UnicodeDecodeError:
            raise ValueError('invalid non-ASCII bytes; use unicode string instead: %r' % raw_bytes)
        return _string_element(string, ctx)