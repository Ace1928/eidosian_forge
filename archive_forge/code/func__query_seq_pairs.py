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
@classmethod
def _query_seq_pairs(cls, quoter, pairs):
    for key, val in pairs:
        if isinstance(val, (list, tuple)):
            for v in val:
                yield (quoter(key) + '=' + quoter(cls._query_var(v)))
        else:
            yield (quoter(key) + '=' + quoter(cls._query_var(val)))