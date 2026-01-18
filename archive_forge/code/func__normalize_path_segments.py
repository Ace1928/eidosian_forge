import functools
import math
import warnings
from collections.abc import Mapping, Sequence
from contextlib import suppress
from ipaddress import ip_address
from urllib.parse import SplitResult, parse_qsl, quote, urljoin, urlsplit, urlunsplit
import idna
from multidict import MultiDict, MultiDictProxy
from ._quoting import _Quoter, _Unquoter
def _normalize_path_segments(segments):
    """Drop '.' and '..' from a sequence of str segments"""
    resolved_path = []
    for seg in segments:
        if seg == '..':
            with suppress(IndexError):
                resolved_path.pop()
        elif seg != '.':
            resolved_path.append(seg)
    if segments and segments[-1] in ('.', '..'):
        resolved_path.append('')
    return resolved_path