import base64
import contextlib
import socket
import struct
import time
from typing import Any, Dict, Optional, Tuple, Union
import dns.asyncbackend
import dns.exception
import dns.inet
import dns.message
import dns.name
import dns.quic
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.transaction
from dns._asyncbackend import NullContext
from dns.query import (
def _source_tuple(af, address, port):
    if address or port:
        if address is None:
            if af == socket.AF_INET:
                address = '0.0.0.0'
            elif af == socket.AF_INET6:
                address = '::'
            else:
                raise NotImplementedError(f'unknown address family {af}')
        return (address, port)
    else:
        return None