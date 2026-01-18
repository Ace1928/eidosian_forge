import sys
import os
from collections import namedtuple
from enum import Enum as _Enum, IntEnum as _IntEnum, IntFlag as _IntFlag
from enum import _simple_enum
import _ssl             # if we can't import it, let the error propagate
from _ssl import OPENSSL_VERSION_NUMBER, OPENSSL_VERSION_INFO, OPENSSL_VERSION
from _ssl import _SSLContext, MemoryBIO, SSLSession
from _ssl import (
from _ssl import txt2obj as _txt2obj, nid2obj as _nid2obj
from _ssl import RAND_status, RAND_add, RAND_bytes, RAND_pseudo_bytes
from _ssl import (
from _ssl import _DEFAULT_CIPHERS, _OPENSSL_API_VERSION
from socket import socket, SOCK_STREAM, create_connection
from socket import SOL_SOCKET, SO_TYPE, _GLOBAL_DEFAULT_TIMEOUT
import socket as _socket
import base64        # for DER-to-PEM translation
import errno
import warnings
def _inet_paton(ipname):
    """Try to convert an IP address to packed binary form

    Supports IPv4 addresses on all platforms and IPv6 on platforms with IPv6
    support.
    """
    try:
        addr = _socket.inet_aton(ipname)
    except OSError:
        pass
    else:
        if _socket.inet_ntoa(addr) == ipname:
            return addr
        else:
            raise ValueError('{!r} is not a quad-dotted IPv4 address.'.format(ipname))
    try:
        return _socket.inet_pton(_socket.AF_INET6, ipname)
    except OSError:
        raise ValueError('{!r} is neither an IPv4 nor an IP6 address.'.format(ipname))
    except AttributeError:
        pass
    raise ValueError('{!r} is not an IPv4 address.'.format(ipname))