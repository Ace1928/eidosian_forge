import time
from oslo_log import log as logging
from os_win import _utils
import os_win.conf
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
def _get_job_error_details(self, job):
    try:
        return job.GetErrorEx()
    except Exception:
        LOG.error("Could not get job '%s' error details.", job.InstanceID)