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
def _tls_handshake(s, expiration):
    while True:
        try:
            s.do_handshake()
            return
        except ssl.SSLWantReadError:
            _wait_for_readable(s, expiration)
        except ssl.SSLWantWriteError:
            _wait_for_writable(s, expiration)