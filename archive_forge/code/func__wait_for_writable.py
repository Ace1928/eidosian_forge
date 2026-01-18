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
def _wait_for_writable(s, expiration):
    _wait_for(s, False, True, True, expiration)