import sys
from os_win._i18n import _
class WMIJobFailed(HyperVException):
    msg_fmt = _('WMI job failed with status %(job_state)s. Error summary description: %(error_summ_desc)s. Error description: %(error_desc)s Error code: %(error_code)s.')

    def __init__(self, message=None, **kwargs):
        self.error_code = kwargs.get('error_code', None)
        self.job_state = kwargs.get('job_state', None)
        super(WMIJobFailed, self).__init__(message, **kwargs)