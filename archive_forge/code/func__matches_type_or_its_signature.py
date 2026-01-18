import enum
import io
from typing import Any, Dict, Optional
import dns.immutable
import dns.name
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.renderer
import dns.rrset
def _matches_type_or_its_signature(rdtypes, rdtype, covers):
    return rdtype in rdtypes or (rdtype == dns.rdatatype.RRSIG and covers in rdtypes)