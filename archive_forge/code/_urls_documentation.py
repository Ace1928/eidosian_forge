from __future__ import annotations
import typing
from urllib.parse import parse_qs, unquote
import idna
from ._types import QueryParamTypes, RawURL, URLTypes
from ._urlparse import urlencode, urlparse
from ._utils import primitive_value_to_str

        Note that we use '%20' encoding for spaces, and treat '/' as a safe
        character.

        See https://github.com/encode/httpx/issues/2536 and
        https://docs.python.org/3/library/urllib.parse.html#urllib.parse.urlencode
        