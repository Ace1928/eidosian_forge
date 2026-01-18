import collections
import random
import struct
from typing import Any, List
import dns.exception
import dns.ipv4
import dns.ipv6
import dns.name
import dns.rdata
def _priority_table(items):
    by_priority = collections.defaultdict(list)
    for rdata in items:
        by_priority[rdata._processing_priority()].append(rdata)
    return by_priority