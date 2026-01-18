import time
from oslo_log import log as logging
from os_win import _utils
import os_win.conf
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
@_utils.retry_decorator(exceptions=exceptions.JobTerminateFailed, timeout=timeout, max_retry_count=None)
def _stop_jobs_with_timeout():
    self._stop_jobs(element)