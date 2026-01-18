import errno
import os
from eventlet import patcher
from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
def _write_to_log(self, data):
    if self._stopped.isSet():
        return
    try:
        log_size = self._log_file_handle.tell() + len(data)
        if log_size >= constants.MAX_CONSOLE_LOG_FILE_SIZE:
            self._rotate_logs()
        self._log_file_handle.write(data)
    except Exception:
        self._stopped.set()