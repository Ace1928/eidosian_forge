from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
def _check_bracketed_host(hostname):
    if hostname.startswith('v'):
        if not re.match('\\Av[a-fA-F0-9]+\\..+\\Z', hostname):
            raise ValueError(f'IPvFuture address is invalid')
    else:
        ip = ipaddress.ip_address(hostname)
        if isinstance(ip, ipaddress.IPv4Address):
            raise ValueError(f'An IPv4 address cannot be in brackets')