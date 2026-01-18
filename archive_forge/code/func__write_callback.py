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
def _write_callback(connection_id, data_buffer, data_length_pointer):
    """
    SecureTransport write callback. This is called by ST to request that data
    actually be sent on the network.
    """
    wrapped_socket = None
    try:
        wrapped_socket = _connection_refs.get(connection_id)
        if wrapped_socket is None:
            return SecurityConst.errSSLInternal
        base_socket = wrapped_socket.socket
        bytes_to_write = data_length_pointer[0]
        data = ctypes.string_at(data_buffer, bytes_to_write)
        timeout = wrapped_socket.gettimeout()
        error = None
        sent = 0
        try:
            while sent < bytes_to_write:
                if timeout is None or timeout >= 0:
                    if not util.wait_for_write(base_socket, timeout):
                        raise socket.error(errno.EAGAIN, 'timed out')
                chunk_sent = base_socket.send(data)
                sent += chunk_sent
                data = data[chunk_sent:]
        except socket.error as e:
            error = e.errno
            if error is not None and error != errno.EAGAIN:
                data_length_pointer[0] = sent
                if error == errno.ECONNRESET or error == errno.EPIPE:
                    return SecurityConst.errSSLClosedAbort
                raise
        data_length_pointer[0] = sent
        if sent != bytes_to_write:
            return SecurityConst.errSSLWouldBlock
        return 0
    except Exception as e:
        if wrapped_socket is not None:
            wrapped_socket._exception = e
        return SecurityConst.errSSLInternal