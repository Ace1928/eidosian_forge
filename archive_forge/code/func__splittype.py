from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
def _splittype(url):
    """splittype('type:opaquestring') --> 'type', 'opaquestring'."""
    global _typeprog
    if _typeprog is None:
        _typeprog = re.compile('([^/:]+):(.*)', re.DOTALL)
    match = _typeprog.match(url)
    if match:
        scheme, data = match.groups()
        return (scheme.lower(), data)
    return (None, url)