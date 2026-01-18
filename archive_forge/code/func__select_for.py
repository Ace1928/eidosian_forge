from __future__ import generators
import errno
import select
import socket
import struct
import sys
import time
import dns.exception
import dns.inet
import dns.name
import dns.message
import dns.rcode
import dns.rdataclass
import dns.rdatatype
from ._compat import long, string_types, PY3
def _select_for(fd, readable, writable, error, timeout):
    """Select polling backend."""
    rset, wset, xset = ([], [], [])
    if readable:
        rset = [fd]
    if writable:
        wset = [fd]
    if error:
        xset = [fd]
    if timeout is None:
        rcount, wcount, xcount = select.select(rset, wset, xset)
    else:
        rcount, wcount, xcount = select.select(rset, wset, xset, timeout)
    return bool(rcount or wcount or xcount)