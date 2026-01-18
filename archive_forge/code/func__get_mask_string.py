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
@staticmethod
def _get_mask_string(mask):
    masks = []
    for c in dir(InotifyConstants):
        if c.startswith('IN_') and c not in ['IN_ALL_EVENTS', 'IN_CLOSE', 'IN_MOVE']:
            c_val = getattr(InotifyConstants, c)
            if mask & c_val:
                masks.append(c)
    mask_string = '|'.join(masks)
    return mask_string