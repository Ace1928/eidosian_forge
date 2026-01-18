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
def _compute_tsig_reserve(self) -> int:
    """Compute the size required for the TSIG RR"""
    if not self.tsig:
        return 0
    f = io.BytesIO()
    self.tsig.to_wire(f)
    return len(f.getvalue())