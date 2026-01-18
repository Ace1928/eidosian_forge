import contextlib
import random
import socket
import sys
import threading
import time
import warnings
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse
import dns._ddr
import dns.edns
import dns.exception
import dns.flags
import dns.inet
import dns.ipv4
import dns.ipv6
import dns.message
import dns.name
import dns.nameserver
import dns.query
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.rdtypes.svcbbase
import dns.reversename
import dns.tsig
class CacheStatistics:
    """Cache Statistics"""

    def __init__(self, hits: int=0, misses: int=0) -> None:
        self.hits = hits
        self.misses = misses

    def reset(self) -> None:
        self.hits = 0
        self.misses = 0

    def clone(self) -> 'CacheStatistics':
        return CacheStatistics(self.hits, self.misses)