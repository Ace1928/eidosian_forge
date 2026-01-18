from __future__ import annotations
import collections
import contextlib
import copy
import os
import platform
import socket
import ssl
import sys
import threading
import time
import weakref
from typing import (
import bson
from bson import DEFAULT_CODEC_OPTIONS
from bson.son import SON
from pymongo import __version__, _csot, auth, helpers
from pymongo.client_session import _validate_session_write_concern
from pymongo.common import (
from pymongo.errors import (
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_reauth
from pymongo.lock import _create_lock
from pymongo.monitoring import (
from pymongo.network import command, receive_message
from pymongo.read_preferences import ReadPreference
from pymongo.server_api import _add_to_command
from pymongo.server_type import SERVER_TYPE
from pymongo.socket_checker import SocketChecker
from pymongo.ssl_support import HAS_SNI, SSLError
def _configured_socket(address: _Address, options: PoolOptions) -> Union[socket.socket, _sslConn]:
    """Given (host, port) and PoolOptions, return a configured socket.

    Can raise socket.error, ConnectionFailure, or _CertificateError.

    Sets socket's SSL and timeout options.
    """
    sock = _create_connection(address, options)
    ssl_context = options._ssl_context
    if ssl_context is None:
        sock.settimeout(options.socket_timeout)
        return sock
    host = address[0]
    try:
        if HAS_SNI:
            ssl_sock = ssl_context.wrap_socket(sock, server_hostname=host)
        else:
            ssl_sock = ssl_context.wrap_socket(sock)
    except _CertificateError:
        sock.close()
        raise
    except (OSError, SSLError) as exc:
        sock.close()
        details = _get_timeout_details(options)
        _raise_connection_failure(address, exc, 'SSL handshake failed: ', timeout_details=details)
    if ssl_context.verify_mode and (not ssl_context.check_hostname) and (not options.tls_allow_invalid_hostnames):
        try:
            ssl.match_hostname(ssl_sock.getpeercert(), hostname=host)
        except _CertificateError:
            ssl_sock.close()
            raise
    ssl_sock.settimeout(options.socket_timeout)
    return ssl_sock