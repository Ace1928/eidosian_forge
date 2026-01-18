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
def _add_watch(self, path, mask):
    """
        Adds a watch for the given path to monitor events specified by the
        mask.

        :param path:
            Path to monitor
        :param mask:
            Event bit mask.
        """
    wd = inotify_add_watch(self._inotify_fd, path, mask)
    if wd == -1:
        Inotify._raise_error()
    self._wd_for_path[path] = wd
    self._path_for_wd[wd] = path
    return wd