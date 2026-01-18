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
@_simple_enum(_IntEnum)
class _TLSMessageType:
    """Message types (handshake protocol)

    See RFC 8446, section B.3
    """
    HELLO_REQUEST = 0
    CLIENT_HELLO = 1
    SERVER_HELLO = 2
    HELLO_VERIFY_REQUEST = 3
    NEWSESSION_TICKET = 4
    END_OF_EARLY_DATA = 5
    HELLO_RETRY_REQUEST = 6
    ENCRYPTED_EXTENSIONS = 8
    CERTIFICATE = 11
    SERVER_KEY_EXCHANGE = 12
    CERTIFICATE_REQUEST = 13
    SERVER_DONE = 14
    CERTIFICATE_VERIFY = 15
    CLIENT_KEY_EXCHANGE = 16
    FINISHED = 20
    CERTIFICATE_URL = 21
    CERTIFICATE_STATUS = 22
    SUPPLEMENTAL_DATA = 23
    KEY_UPDATE = 24
    NEXT_PROTO = 67
    MESSAGE_HASH = 254
    CHANGE_CIPHER_SPEC = 257