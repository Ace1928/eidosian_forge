from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
def _splitport(host):
    """splitport('host:port') --> 'host', 'port'."""
    global _portprog
    if _portprog is None:
        _portprog = re.compile('(.*):([0-9]*)', re.DOTALL)
    match = _portprog.fullmatch(host)
    if match:
        host, port = match.groups()
        if port:
            return (host, port)
    return (host, None)