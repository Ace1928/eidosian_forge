import time
from oslo_log import log as logging
from os_win import _utils
import os_win.conf
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
def _stop_jobs(self, element):
    pending_jobs = self._get_pending_jobs_affecting_element(element)
    for job in pending_jobs:
        job_details = self._get_job_details(job, extended=True)
        try:
            if not job.Cancellable:
                LOG.debug('Got request to terminate non-cancelable job: %s.', job_details)
                continue
            job.RequestStateChange(self._KILL_JOB_STATE_CHANGE_REQUEST)
        except exceptions.x_wmi as ex:
            if not _utils._is_not_found_exc(ex):
                LOG.debug('Failed to stop job. Exception: %s. Job details: %s.', ex, job_details)
    pending_jobs = self._get_pending_jobs_affecting_element(element)
    if pending_jobs:
        pending_job_details = [self._get_job_details(job, extended=True) for job in pending_jobs]
        LOG.debug('Attempted to terminate jobs affecting element %(element)s but %(pending_count)s jobs are still pending: %(pending_jobs)s.', dict(element=element, pending_count=len(pending_jobs), pending_jobs=pending_job_details))
        raise exceptions.JobTerminateFailed()