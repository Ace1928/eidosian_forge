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
@inline
@cfunc
def _check_closed_deep(s: Socket) -> bint:
    """thorough check of whether the socket has been closed,
    even if by another entity (e.g. ctx.destroy).

    Only used by the `closed` property.

    returns True if closed, False otherwise
    """
    rc: C.int
    errno: C.int
    stype = declare(C.int)
    sz: size_t = sizeof(int)
    if s._closed:
        return True
    else:
        rc = zmq_getsockopt(s.handle, ZMQ_TYPE, cast(p_void, address(stype)), address(sz))
        if rc < 0:
            errno = zmq_errno()
            if errno == ENOTSOCK:
                s._closed = True
                return True
            elif errno == ZMQ_ETERM:
                return False
        else:
            _check_rc(rc)
    return False