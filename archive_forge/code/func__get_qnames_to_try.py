import contextlib
import random
import socket
import sys
import threading
import time
import warnings
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse
import dns._ddr
import dns.edns
import dns.exception
import dns.flags
import dns.inet
import dns.ipv4
import dns.ipv6
import dns.message
import dns.name
import dns.nameserver
import dns.query
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.rdtypes.svcbbase
import dns.reversename
import dns.tsig
def _get_qnames_to_try(self, qname: dns.name.Name, search: Optional[bool]) -> List[dns.name.Name]:
    if search is None:
        search = self.use_search_by_default
    qnames_to_try = []
    if qname.is_absolute():
        qnames_to_try.append(qname)
    else:
        abs_qname = qname.concatenate(dns.name.root)
        if search:
            if len(self.search) > 0:
                search_list = self.search[:]
            elif self.domain != dns.name.root and self.domain is not None:
                search_list = [self.domain]
            else:
                search_list = []
            if self.ndots is None:
                ndots = 1
            else:
                ndots = self.ndots
            for suffix in search_list:
                qnames_to_try.append(qname + suffix)
            if len(qname) > ndots:
                qnames_to_try.insert(0, abs_qname)
            else:
                qnames_to_try.append(abs_qname)
        else:
            qnames_to_try.append(abs_qname)
    return qnames_to_try