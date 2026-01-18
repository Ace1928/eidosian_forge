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
@staticmethod
def _make_tsig(keyname, algorithm, time_signed, fudge, mac, original_id, error, other):
    tsig = dns.rdtypes.ANY.TSIG.TSIG(dns.rdataclass.ANY, dns.rdatatype.TSIG, algorithm, time_signed, fudge, mac, original_id, error, other)
    return dns.rrset.from_rdata(keyname, 0, tsig)