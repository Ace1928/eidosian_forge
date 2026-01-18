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
def _upper_dollarize(s):
    s = s.upper()
    if not s.startswith('$'):
        s = '$' + s
    return s