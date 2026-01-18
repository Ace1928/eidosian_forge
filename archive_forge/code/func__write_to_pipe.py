import errno
import os
from eventlet import patcher
from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
def _write_to_pipe(self):
    self._start_io_worker(self._ioutils.write, self._w_buffer, self._w_overlapped, self._w_completion_routine, self._get_data_to_write)