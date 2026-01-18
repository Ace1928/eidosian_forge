import collections
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union
import dns.exception
import dns.name
import dns.node
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.rrset
import dns.serial
import dns.ttl
def _ensure_immutable_rdataset(rdataset):
    if rdataset is None or isinstance(rdataset, dns.rdataset.ImmutableRdataset):
        return rdataset
    return dns.rdataset.ImmutableRdataset(rdataset)