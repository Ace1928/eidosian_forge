import time
from oslo_log import log as logging
from os_win import _utils
import os_win.conf
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
def _get_job_details(self, job, extended=False):
    basic_details = ['InstanceID', 'Description', 'ElementName', 'JobStatus', 'ElapsedTime', 'Cancellable', 'JobType', 'Owner', 'PercentComplete']
    extended_details = ['JobState', 'StatusDescriptions', 'OperationalStatus', 'TimeSubmitted', 'UntilTime', 'TimeOfLastStateChange', 'DetailedStatus', 'LocalOrUtcTime', 'ErrorCode', 'ErrorDescription', 'ErrorSummaryDescription']
    fields = list(basic_details)
    details = {}
    if extended:
        fields += extended_details
        err_details = self._get_job_error_details(job)
        details['RawErrors'] = err_details
    for field in fields:
        try:
            details[field] = getattr(job, field)
        except AttributeError:
            continue
    return details