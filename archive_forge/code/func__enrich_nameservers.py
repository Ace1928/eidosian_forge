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
@classmethod
def _enrich_nameservers(cls, nameservers: Sequence[Union[str, dns.nameserver.Nameserver]], nameserver_ports: Dict[str, int], default_port: int) -> List[dns.nameserver.Nameserver]:
    enriched_nameservers = []
    if isinstance(nameservers, list):
        for nameserver in nameservers:
            enriched_nameserver: dns.nameserver.Nameserver
            if isinstance(nameserver, dns.nameserver.Nameserver):
                enriched_nameserver = nameserver
            elif dns.inet.is_address(nameserver):
                port = nameserver_ports.get(nameserver, default_port)
                enriched_nameserver = dns.nameserver.Do53Nameserver(nameserver, port)
            else:
                try:
                    if urlparse(nameserver).scheme != 'https':
                        raise NotImplementedError
                except Exception:
                    raise ValueError(f'nameserver {nameserver} is not a dns.nameserver.Nameserver instance or text form, IP address, nor a valid https URL')
                enriched_nameserver = dns.nameserver.DoHNameserver(nameserver)
            enriched_nameservers.append(enriched_nameserver)
    else:
        raise ValueError('nameservers must be a list or tuple (not a {})'.format(type(nameservers)))
    return enriched_nameservers