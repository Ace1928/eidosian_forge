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
def _wait_for(fd, readable, writable, error, expiration):
    done = False
    while not done:
        if expiration is None:
            timeout = None
        else:
            timeout = expiration - time.time()
            if timeout <= 0.0:
                raise dns.exception.Timeout
        try:
            if not _polling_backend(fd, readable, writable, error, timeout):
                raise dns.exception.Timeout
        except select_error as e:
            if e.args[0] != errno.EINTR:
                raise e
        done = True