from __future__ import annotations
import io
import mimetypes
import os
import pkgutil
import re
import sys
import typing as t
import unicodedata
from datetime import datetime
from time import time
from urllib.parse import quote
from zlib import adler32
from markupsafe import escape
from ._internal import _DictAccessorProperty
from ._internal import _missing
from ._internal import _TAccessorValue
from .datastructures import Headers
from .exceptions import NotFound
from .exceptions import RequestedRangeNotSatisfiable
from .security import safe_join
from .wsgi import wrap_file
class header_property(_DictAccessorProperty[_TAccessorValue]):
    """Like `environ_property` but for headers."""

    def lookup(self, obj: Request | Response) -> Headers:
        return obj.headers