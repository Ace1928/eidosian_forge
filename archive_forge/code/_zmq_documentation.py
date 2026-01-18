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

        Receive a message.

        With flags=NOBLOCK, this raises :class:`ZMQError` if no messages have
        arrived; otherwise, this waits until a message arrives.
        See :class:`Poller` for more general non-blocking I/O.

        Parameters
        ----------
        flags : int
            0 or NOBLOCK.
        copy : bool
            Should the message be received in a copying or non-copying manner?
            If False a Frame object is returned, if True a string copy of
            message is returned.
        track : bool
            Should the message be tracked for notification that ZMQ has
            finished with it? (ignored if copy=True)

        Returns
        -------
        msg : bytes or Frame
            The received message frame.  If `copy` is False, then it will be a Frame,
            otherwise it will be bytes.

        Raises
        ------
        ZMQError
            for any of the reasons zmq_msg_recv might fail (including if
            NOBLOCK is set and no new messages have arrived).
        