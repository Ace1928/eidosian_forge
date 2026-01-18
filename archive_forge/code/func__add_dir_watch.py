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
def _add_dir_watch(self, path, recursive, mask):
    """
        Adds a watch (optionally recursively) for the given directory path
        to monitor events specified by the mask.

        :param path:
            Path to monitor
        :param recursive:
            ``True`` to monitor recursively.
        :param mask:
            Event bit mask.
        """
    if not os.path.isdir(path):
        raise OSError('Path is not a directory')
    self._add_watch(path, mask)
    if recursive:
        for root, dirnames, _ in os.walk(path):
            for dirname in dirnames:
                full_path = os.path.join(root, dirname)
                if os.path.islink(full_path):
                    continue
                self._add_watch(full_path, mask)