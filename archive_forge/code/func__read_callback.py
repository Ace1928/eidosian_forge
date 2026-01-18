from __future__ import absolute_import
import contextlib
import ctypes
import errno
import os.path
import shutil
import socket
import ssl
import struct
import threading
import weakref
import six
from .. import util
from ..util.ssl_ import PROTOCOL_TLS_CLIENT
from ._securetransport.bindings import CoreFoundation, Security, SecurityConst
from ._securetransport.low_level import (
def _read_callback(connection_id, data_buffer, data_length_pointer):
    """
    SecureTransport read callback. This is called by ST to request that data
    be returned from the socket.
    """
    wrapped_socket = None
    try:
        wrapped_socket = _connection_refs.get(connection_id)
        if wrapped_socket is None:
            return SecurityConst.errSSLInternal
        base_socket = wrapped_socket.socket
        requested_length = data_length_pointer[0]
        timeout = wrapped_socket.gettimeout()
        error = None
        read_count = 0
        try:
            while read_count < requested_length:
                if timeout is None or timeout >= 0:
                    if not util.wait_for_read(base_socket, timeout):
                        raise socket.error(errno.EAGAIN, 'timed out')
                remaining = requested_length - read_count
                buffer = (ctypes.c_char * remaining).from_address(data_buffer + read_count)
                chunk_size = base_socket.recv_into(buffer, remaining)
                read_count += chunk_size
                if not chunk_size:
                    if not read_count:
                        return SecurityConst.errSSLClosedGraceful
                    break
        except socket.error as e:
            error = e.errno
            if error is not None and error != errno.EAGAIN:
                data_length_pointer[0] = read_count
                if error == errno.ECONNRESET or error == errno.EPIPE:
                    return SecurityConst.errSSLClosedAbort
                raise
        data_length_pointer[0] = read_count
        if read_count != requested_length:
            return SecurityConst.errSSLWouldBlock
        return 0
    except Exception as e:
        if wrapped_socket is not None:
            wrapped_socket._exception = e
        return SecurityConst.errSSLInternal