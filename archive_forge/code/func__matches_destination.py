import base64
import contextlib
import enum
import errno
import os
import os.path
import selectors
import socket
import struct
import time
from typing import Any, Dict, Optional, Tuple, Union
import dns._features
import dns.exception
import dns.inet
import dns.message
import dns.name
import dns.quic
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.serial
import dns.transaction
import dns.tsig
import dns.xfr
def _matches_destination(af, from_address, destination, ignore_unexpected):
    if not destination:
        return True
    if _addresses_equal(af, from_address, destination) or (dns.inet.is_multicast(destination[0]) and from_address[1:] == destination[1:]):
        return True
    elif ignore_unexpected:
        return False
    raise UnexpectedSource(f'got a response from {from_address} instead of {destination}')