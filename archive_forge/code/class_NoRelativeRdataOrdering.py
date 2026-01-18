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
class NoRelativeRdataOrdering(dns.exception.DNSException):
    """An attempt was made to do an ordered comparison of one or more
    rdata with relative names.  The only reliable way of sorting rdata
    is to use non-relativized rdata.

    """