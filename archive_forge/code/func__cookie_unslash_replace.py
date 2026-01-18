from __future__ import annotations
import re
import typing as t
from datetime import datetime
from .._internal import _dt_as_utc
from ..http import generate_etag
from ..http import parse_date
from ..http import parse_etags
from ..http import parse_if_range_header
from ..http import unquote_etag
from .. import datastructures as ds
def _cookie_unslash_replace(m: t.Match[bytes]) -> bytes:
    v = m.group(1)
    if len(v) == 1:
        return v
    return int(v, 8).to_bytes(1, 'big')