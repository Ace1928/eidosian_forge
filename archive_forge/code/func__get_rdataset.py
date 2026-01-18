import re
import sys
from typing import Any, Iterable, List, Optional, Set, Tuple, Union
import dns.exception
import dns.grange
import dns.name
import dns.node
import dns.rdata
import dns.rdataclass
import dns.rdatatype
import dns.rdtypes.ANY.SOA
import dns.rrset
import dns.tokenizer
import dns.transaction
import dns.ttl
def _get_rdataset(self, name, rdtype, covers):
    return self.rdatasets.get((name, rdtype, covers))