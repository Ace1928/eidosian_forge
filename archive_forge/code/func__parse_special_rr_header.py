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
def _parse_special_rr_header(self, section, count, position, name, rdclass, rdtype):
    if rdtype == dns.rdatatype.OPT:
        if section != MessageSection.ADDITIONAL or self.opt or name != dns.name.root:
            raise BadEDNS
    elif rdtype == dns.rdatatype.TSIG:
        if section != MessageSection.ADDITIONAL or rdclass != dns.rdatatype.ANY or position != count - 1:
            raise BadTSIG
    return (rdclass, rdtype, None, False)