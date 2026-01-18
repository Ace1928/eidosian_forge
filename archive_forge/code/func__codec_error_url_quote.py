from __future__ import annotations
import codecs
import re
import typing as t
from urllib.parse import quote
from urllib.parse import unquote
from urllib.parse import urlencode
from urllib.parse import urlsplit
from urllib.parse import urlunsplit
from .datastructures import iter_multi_items
def _codec_error_url_quote(e: UnicodeError) -> tuple[str, int]:
    """Used in :func:`uri_to_iri` after unquoting to re-quote any
    invalid bytes.
    """
    out = quote(e.object[e.start:e.end], safe='')
    return (out, e.end)