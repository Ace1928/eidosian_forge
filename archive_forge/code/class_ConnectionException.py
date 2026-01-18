from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
class ConnectionException(XcffibException):
    REASONS = {lib.XCB_CONN_ERROR: 'xcb connection errors because of socket, pipe and other stream errors.', lib.XCB_CONN_CLOSED_EXT_NOTSUPPORTED: 'xcb connection shutdown because extension not supported', lib.XCB_CONN_CLOSED_MEM_INSUFFICIENT: 'malloc(), calloc() and realloc() error upon failure, for eg ENOMEM', lib.XCB_CONN_CLOSED_REQ_LEN_EXCEED: 'Connection closed, exceeding request length that server accepts.', lib.XCB_CONN_CLOSED_PARSE_ERR: 'Connection closed, error during parsing display string.', lib.XCB_CONN_CLOSED_INVALID_SCREEN: 'Connection closed because the server does not have a screen matching the display.', lib.XCB_CONN_CLOSED_FDPASSING_FAILED: 'Connection closed because some FD passing operation failed'}

    def __init__(self, err):
        XcffibException.__init__(self, self.REASONS.get(err, 'Unknown connection error.'))