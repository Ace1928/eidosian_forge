import base64
import binascii
import inspect
import io
import itertools
import random
from importlib import import_module
from typing import Any, Dict, Optional, Tuple, Union
import dns.exception
import dns.immutable
import dns.ipv4
import dns.ipv6
import dns.name
import dns.rdataclass
import dns.rdatatype
import dns.tokenizer
import dns.ttl
import dns.wire
@classmethod
def _as_ipv4_address(cls, value):
    if isinstance(value, str):
        return dns.ipv4.canonicalize(value)
    elif isinstance(value, bytes):
        return dns.ipv4.inet_ntoa(value)
    else:
        raise ValueError('not an IPv4 address')