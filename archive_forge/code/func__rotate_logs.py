import errno
import os
from eventlet import patcher
from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
def _rotate_logs(self):
    self._log_file_handle.flush()
    self._log_file_handle.close()
    log_archive_path = self._log_file_path + '.1'
    if os.path.exists(log_archive_path):
        self._retry_if_file_in_use(os.remove, log_archive_path)
    self._retry_if_file_in_use(os.rename, self._log_file_path, log_archive_path)
    self._log_file_handle = open(self._log_file_path, 'ab', 1)