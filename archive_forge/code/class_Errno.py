from __future__ import annotations
import errno
import sys
from enum import Enum, IntEnum, IntFlag
class Errno(IntEnum):
    """libzmq error codes

    .. versionadded:: 23
    """
    EAGAIN = errno.EAGAIN
    EFAULT = errno.EFAULT
    EINVAL = errno.EINVAL
    if sys.platform.startswith('win'):
        ENOTSUP = 129
        EPROTONOSUPPORT = 135
        ENOBUFS = 119
        ENETDOWN = 116
        EADDRINUSE = 100
        EADDRNOTAVAIL = 101
        ECONNREFUSED = 107
        EINPROGRESS = 112
        ENOTSOCK = 128
        EMSGSIZE = 115
        EAFNOSUPPORT = 102
        ENETUNREACH = 118
        ECONNABORTED = 106
        ECONNRESET = 108
        ENOTCONN = 126
        ETIMEDOUT = 138
        EHOSTUNREACH = 110
        ENETRESET = 117
    else:
        ENOTSUP = getattr(errno, 'ENOTSUP', _HAUSNUMERO + 1)
        EPROTONOSUPPORT = getattr(errno, 'EPROTONOSUPPORT', _HAUSNUMERO + 2)
        ENOBUFS = getattr(errno, 'ENOBUFS', _HAUSNUMERO + 3)
        ENETDOWN = getattr(errno, 'ENETDOWN', _HAUSNUMERO + 4)
        EADDRINUSE = getattr(errno, 'EADDRINUSE', _HAUSNUMERO + 5)
        EADDRNOTAVAIL = getattr(errno, 'EADDRNOTAVAIL', _HAUSNUMERO + 6)
        ECONNREFUSED = getattr(errno, 'ECONNREFUSED', _HAUSNUMERO + 7)
        EINPROGRESS = getattr(errno, 'EINPROGRESS', _HAUSNUMERO + 8)
        ENOTSOCK = getattr(errno, 'ENOTSOCK', _HAUSNUMERO + 9)
        EMSGSIZE = getattr(errno, 'EMSGSIZE', _HAUSNUMERO + 10)
        EAFNOSUPPORT = getattr(errno, 'EAFNOSUPPORT', _HAUSNUMERO + 11)
        ENETUNREACH = getattr(errno, 'ENETUNREACH', _HAUSNUMERO + 12)
        ECONNABORTED = getattr(errno, 'ECONNABORTED', _HAUSNUMERO + 13)
        ECONNRESET = getattr(errno, 'ECONNRESET', _HAUSNUMERO + 14)
        ENOTCONN = getattr(errno, 'ENOTCONN', _HAUSNUMERO + 15)
        ETIMEDOUT = getattr(errno, 'ETIMEDOUT', _HAUSNUMERO + 16)
        EHOSTUNREACH = getattr(errno, 'EHOSTUNREACH', _HAUSNUMERO + 17)
        ENETRESET = getattr(errno, 'ENETRESET', _HAUSNUMERO + 18)
    EFSM = _HAUSNUMERO + 51
    ENOCOMPATPROTO = _HAUSNUMERO + 52
    ETERM = _HAUSNUMERO + 53
    EMTHREAD = _HAUSNUMERO + 54