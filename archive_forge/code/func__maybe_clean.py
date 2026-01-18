import socket
import sys
import time
import random
import dns.exception
import dns.flags
import dns.ipv4
import dns.ipv6
import dns.message
import dns.name
import dns.query
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.reversename
import dns.tsig
from ._compat import xrange, string_types
def _maybe_clean(self):
    """Clean the cache if it's time to do so."""
    now = time.time()
    if self.next_cleaning <= now:
        keys_to_delete = []
        for k, v in self.data.items():
            if v.expiration <= now:
                keys_to_delete.append(k)
        for k in keys_to_delete:
            del self.data[k]
        now = time.time()
        self.next_cleaning = now + self.cleaning_interval