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
def _compute_opt_reserve(self) -> int:
    """Compute the size required for the OPT RR, padding excluded"""
    if not self.opt:
        return 0
    size = 11
    for option in self.opt[0].options:
        wire = option.to_wire()
        size += len(wire) + 4
    if self.pad:
        size += 4
    return size