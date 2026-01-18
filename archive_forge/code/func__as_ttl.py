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
def _as_ttl(cls, value):
    if isinstance(value, int):
        return cls._as_int(value, 0, dns.ttl.MAX_TTL)
    elif isinstance(value, str):
        return dns.ttl.from_text(value)
    else:
        raise ValueError('not a TTL')