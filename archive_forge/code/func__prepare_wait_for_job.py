from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import jobutils
def _prepare_wait_for_job(self, state=_FAKE_JOB_STATUS_BAD, error_code=0):
    mock_job = mock.MagicMock()
    mock_job.JobState = state
    mock_job.ErrorCode = error_code
    mock_job.Description = self._FAKE_JOB_DESCRIPTION
    mock_job.ElapsedTime = self._FAKE_ELAPSED_TIME
    wmi_patcher = mock.patch.object(jobutils.JobUtils, '_get_wmi_obj')
    mock_wmi = wmi_patcher.start()
    self.addCleanup(wmi_patcher.stop)
    mock_wmi.return_value = mock_job
    return mock_job