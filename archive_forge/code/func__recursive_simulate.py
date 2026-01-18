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
def _recursive_simulate(src_path):
    events = []
    for root, dirnames, filenames in os.walk(src_path):
        for dirname in dirnames:
            try:
                full_path = os.path.join(root, dirname)
                wd_dir = self._add_watch(full_path, self._event_mask)
                e = InotifyEvent(wd_dir, InotifyConstants.IN_CREATE | InotifyConstants.IN_ISDIR, 0, dirname, full_path)
                events.append(e)
            except OSError:
                pass
        for filename in filenames:
            full_path = os.path.join(root, filename)
            wd_parent_dir = self._wd_for_path[os.path.dirname(full_path)]
            e = InotifyEvent(wd_parent_dir, InotifyConstants.IN_CREATE, 0, filename, full_path)
            events.append(e)
    return events