import errno
import os
from eventlet import patcher
from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
def _cancel_io(self):
    if self._pipe_handle:
        self._ioutils.cancel_io(self._pipe_handle, self._r_overlapped, ignore_invalid_handle=True)
        self._ioutils.cancel_io(self._pipe_handle, self._w_overlapped, ignore_invalid_handle=True)