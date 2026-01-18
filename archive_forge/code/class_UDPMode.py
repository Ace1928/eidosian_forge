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
class UDPMode(enum.IntEnum):
    """How should UDP be used in an IXFR from :py:func:`inbound_xfr()`?

    NEVER means "never use UDP; always use TCP"
    TRY_FIRST means "try to use UDP but fall back to TCP if needed"
    ONLY means "raise ``dns.xfr.UseTCP`` if trying UDP does not succeed"
    """
    NEVER = 0
    TRY_FIRST = 1
    ONLY = 2