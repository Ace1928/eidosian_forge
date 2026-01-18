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
def _encode_host(cls, host, human=False):
    try:
        ip, sep, zone = host.partition('%')
        ip = ip_address(ip)
    except ValueError:
        host = host.lower()
        if human or host.isascii():
            return host
        host = _idna_encode(host)
    else:
        host = ip.compressed
        if sep:
            host += '%' + zone
        if ip.version == 6:
            host = '[' + host + ']'
    return host