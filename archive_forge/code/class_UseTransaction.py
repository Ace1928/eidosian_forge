import collections
import threading
from typing import Callable, Deque, Optional, Set, Union
import dns.exception
import dns.immutable
import dns.name
import dns.node
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.rdtypes.ANY.SOA
import dns.zone
class UseTransaction(dns.exception.DNSException):
    """To alter a versioned zone, use a transaction."""