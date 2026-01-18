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
class MessageSection(dns.enum.IntEnum):
    """Message sections"""
    QUESTION = 0
    ANSWER = 1
    AUTHORITY = 2
    ADDITIONAL = 3

    @classmethod
    def _maximum(cls):
        return 3