from __future__ import annotations
import time
import warnings
from threading import Event
from weakref import ref
import cython as C
from cython import (
from cython.cimports.cpython import (
from cython.cimports.libc.errno import EAGAIN, EINTR, ENAMETOOLONG, ENOENT, ENOTSOCK
from cython.cimports.libc.stdint import uint32_t
from cython.cimports.libc.stdio import fprintf
from cython.cimports.libc.stdio import stderr as cstderr
from cython.cimports.libc.stdlib import free, malloc
from cython.cimports.libc.string import memcpy
from cython.cimports.zmq.backend.cython._externs import (
from cython.cimports.zmq.backend.cython.libzmq import (
from cython.cimports.zmq.backend.cython.libzmq import zmq_errno as _zmq_errno
from cython.cimports.zmq.backend.cython.libzmq import zmq_poll as zmq_poll_c
from cython.cimports.zmq.utils.buffers import asbuffer_r
import zmq
from zmq.constants import SocketOption, _OptType
from zmq.error import InterruptedSystemCall, ZMQError, _check_version
@cfunc
@inline
def _send_copy(handle: p_void, buf, flags: C.int=0):
    """Send a message on this socket by copying its content."""
    rc: C.int
    msg = declare(zmq_msg_t)
    c_bytes = declare(p_char)
    c_bytes_len: Py_ssize_t = 0
    asbuffer_r(buf, cast(pointer(p_void), address(c_bytes)), address(c_bytes_len))
    rc = zmq_msg_init_size(address(msg), c_bytes_len)
    _check_rc(rc)
    while True:
        with nogil:
            memcpy(zmq_msg_data(address(msg)), c_bytes, zmq_msg_size(address(msg)))
            rc = zmq_msg_send(address(msg), handle, flags)
        try:
            _check_rc(rc)
        except InterruptedSystemCall:
            continue
        except Exception:
            zmq_msg_close(address(msg))
            raise
        else:
            rc = zmq_msg_close(address(msg))
            _check_rc(rc)
            break