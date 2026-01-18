from __future__ import absolute_import
import re
from collections import namedtuple
from ..exceptions import LocationParseError
from ..packages import six
def _encode_target(target):
    """Percent-encodes a request target so that there are no invalid characters"""
    path, query = TARGET_RE.match(target).groups()
    target = _encode_invalid_chars(path, PATH_CHARS)
    query = _encode_invalid_chars(query, QUERY_CHARS)
    if query is not None:
        target += '?' + query
    return target