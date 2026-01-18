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
def _as_uint32(cls, value):
    if not isinstance(value, int):
        raise ValueError('not an integer')
    if value < 0 or value > 4294967295:
        raise ValueError('not a uint32')
    return value