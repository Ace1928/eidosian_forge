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
def _wordbreak(data, chunksize=_chunksize, separator=b' '):
    """Break a binary string into chunks of chunksize characters separated by
    a space.
    """
    if not chunksize:
        return data.decode()
    return separator.join([data[i:i + chunksize] for i in range(0, len(data), chunksize)]).decode()