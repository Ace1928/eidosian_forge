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
class InotifyEvent(object):
    """
    Inotify event struct wrapper.

    :param wd:
        Watch descriptor
    :param mask:
        Event mask
    :param cookie:
        Event cookie
    :param name:
        Event name.
    :param src_path:
        Event source path
    """

    def __init__(self, wd, mask, cookie, name, src_path):
        self._wd = wd
        self._mask = mask
        self._cookie = cookie
        self._name = name
        self._src_path = src_path

    @property
    def src_path(self):
        return self._src_path

    @property
    def wd(self):
        return self._wd

    @property
    def mask(self):
        return self._mask

    @property
    def cookie(self):
        return self._cookie

    @property
    def name(self):
        return self._name

    @property
    def is_modify(self):
        return self._mask & InotifyConstants.IN_MODIFY > 0

    @property
    def is_close_write(self):
        return self._mask & InotifyConstants.IN_CLOSE_WRITE > 0

    @property
    def is_close_nowrite(self):
        return self._mask & InotifyConstants.IN_CLOSE_NOWRITE > 0

    @property
    def is_access(self):
        return self._mask & InotifyConstants.IN_ACCESS > 0

    @property
    def is_delete(self):
        return self._mask & InotifyConstants.IN_DELETE > 0

    @property
    def is_delete_self(self):
        return self._mask & InotifyConstants.IN_DELETE_SELF > 0

    @property
    def is_create(self):
        return self._mask & InotifyConstants.IN_CREATE > 0

    @property
    def is_moved_from(self):
        return self._mask & InotifyConstants.IN_MOVED_FROM > 0

    @property
    def is_moved_to(self):
        return self._mask & InotifyConstants.IN_MOVED_TO > 0

    @property
    def is_move(self):
        return self._mask & InotifyConstants.IN_MOVE > 0

    @property
    def is_move_self(self):
        return self._mask & InotifyConstants.IN_MOVE_SELF > 0

    @property
    def is_attrib(self):
        return self._mask & InotifyConstants.IN_ATTRIB > 0

    @property
    def is_ignored(self):
        return self._mask & InotifyConstants.IN_IGNORED > 0

    @property
    def is_directory(self):
        return self.is_delete_self or self.is_move_self or self._mask & InotifyConstants.IN_ISDIR > 0

    @property
    def key(self):
        return (self._src_path, self._wd, self._mask, self._cookie, self._name)

    def __eq__(self, inotify_event):
        return self.key == inotify_event.key

    def __ne__(self, inotify_event):
        return self.key == inotify_event.key

    def __hash__(self):
        return hash(self.key)

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

    def __repr__(self):
        mask_string = self._get_mask_string(self.mask)
        s = '<InotifyEvent: src_path=%s, wd=%d, mask=%s, cookie=%d, name=%s>'
        return s % (self.src_path, self.wd, mask_string, self.cookie, self.name)