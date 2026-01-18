import copy
import encodings.idna  # type: ignore
import functools
import struct
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
import dns._features
import dns.enum
import dns.exception
import dns.immutable
import dns.wire
class NeedSubdomainOfOrigin(dns.exception.DNSException):
    """An absolute name was provided that is not a subdomain of the specified origin."""