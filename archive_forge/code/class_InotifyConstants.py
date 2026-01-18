from __future__ import with_statement
import os
import errno
import struct
import threading
import ctypes
import ctypes.util
from functools import reduce
from ctypes import c_int, c_char_p, c_uint32
from wandb_watchdog.utils import has_attribute
from wandb_watchdog.utils import UnsupportedLibc
class InotifyConstants(object):
    IN_ACCESS = 1
    IN_MODIFY = 2
    IN_ATTRIB = 4
    IN_CLOSE_WRITE = 8
    IN_CLOSE_NOWRITE = 16
    IN_OPEN = 32
    IN_MOVED_FROM = 64
    IN_MOVED_TO = 128
    IN_CREATE = 256
    IN_DELETE = 512
    IN_DELETE_SELF = 1024
    IN_MOVE_SELF = 2048
    IN_CLOSE = IN_CLOSE_WRITE | IN_CLOSE_NOWRITE
    IN_MOVE = IN_MOVED_FROM | IN_MOVED_TO
    IN_UNMOUNT = 8192
    IN_Q_OVERFLOW = 16384
    IN_IGNORED = 32768
    IN_ONLYDIR = 16777216
    IN_DONT_FOLLOW = 33554432
    IN_EXCL_UNLINK = 67108864
    IN_MASK_ADD = 536870912
    IN_ISDIR = 1073741824
    IN_ONESHOT = 2147483648
    IN_ALL_EVENTS = reduce(lambda x, y: x | y, [IN_ACCESS, IN_MODIFY, IN_ATTRIB, IN_CLOSE_WRITE, IN_CLOSE_NOWRITE, IN_OPEN, IN_MOVED_FROM, IN_MOVED_TO, IN_DELETE, IN_CREATE, IN_DELETE_SELF, IN_MOVE_SELF])
    IN_CLOEXEC = 33554432
    IN_NONBLOCK = 16384