import time
from oslo_log import log as logging
from os_win import _utils
import os_win.conf
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
def _is_job_completed(self, job):
    return job.JobState in self._completed_job_states