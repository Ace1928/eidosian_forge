import contextlib
import io
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import dns.edns
import dns.entropy
import dns.enum
import dns.exception
import dns.flags
import dns.name
import dns.opcode
import dns.rcode
import dns.rdata
import dns.rdataclass
import dns.rdatatype
import dns.rdtypes.ANY.OPT
import dns.rdtypes.ANY.TSIG
import dns.renderer
import dns.rrset
import dns.tsig
import dns.ttl
import dns.wire
class ChainingResult:
    """The result of a call to dns.message.QueryMessage.resolve_chaining().

    The ``answer`` attribute is the answer RRSet, or ``None`` if it doesn't
    exist.

    The ``canonical_name`` attribute is the canonical name after all
    chaining has been applied (this is the same name as ``rrset.name`` in cases
    where rrset is not ``None``).

    The ``minimum_ttl`` attribute is the minimum TTL, i.e. the TTL to
    use if caching the data.  It is the smallest of all the CNAME TTLs
    and either the answer TTL if it exists or the SOA TTL and SOA
    minimum values for negative answers.

    The ``cnames`` attribute is a list of all the CNAME RRSets followed to
    get to the canonical name.
    """

    def __init__(self, canonical_name: dns.name.Name, answer: Optional[dns.rrset.RRset], minimum_ttl: int, cnames: List[dns.rrset.RRset]):
        self.canonical_name = canonical_name
        self.answer = answer
        self.minimum_ttl = minimum_ttl
        self.cnames = cnames