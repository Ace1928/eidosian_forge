import io
import random
import struct
from typing import Any, Collection, Dict, List, Optional, Union, cast
import dns.exception
import dns.immutable
import dns.name
import dns.rdata
import dns.rdataclass
import dns.rdatatype
import dns.renderer
import dns.set
import dns.ttl
def _rdata_repr(self):

    def maybe_truncate(s):
        if len(s) > 100:
            return s[:100] + '...'
        return s
    return '[%s]' % ', '.join(('<%s>' % maybe_truncate(str(rr)) for rr in self))