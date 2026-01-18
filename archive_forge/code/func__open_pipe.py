import errno
import os
from eventlet import patcher
from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
def _open_pipe(self):
    """Opens a named pipe in overlapped mode for asyncronous I/O."""
    self._ioutils.wait_named_pipe(self._pipe_name)
    self._pipe_handle = self._ioutils.open(self._pipe_name, desired_access=w_const.GENERIC_READ | w_const.GENERIC_WRITE, share_mode=w_const.FILE_SHARE_READ | w_const.FILE_SHARE_WRITE, creation_disposition=w_const.OPEN_EXISTING, flags_and_attributes=w_const.FILE_FLAG_OVERLAPPED)