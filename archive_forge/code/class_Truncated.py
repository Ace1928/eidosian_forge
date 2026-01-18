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
class Truncated(dns.exception.DNSException):
    """The truncated flag is set."""
    supp_kwargs = {'message'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def message(self):
        """As much of the message as could be processed.

        Returns a ``dns.message.Message``.
        """
        return self.kwargs['message']