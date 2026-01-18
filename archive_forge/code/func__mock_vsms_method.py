from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import jobutils
def _mock_vsms_method(self, vsms_method, return_count):
    args = None
    if return_count == 3:
        args = (mock.sentinel.job_path, mock.MagicMock(), self._FAKE_RET_VAL)
    else:
        args = (mock.sentinel.job_path, self._FAKE_RET_VAL)
    vsms_method.return_value = args
    mock_res_setting_data = mock.MagicMock()
    mock_res_setting_data.GetText_.return_value = mock.sentinel.res_data
    mock_res_setting_data.path_.return_value = mock.sentinel.res_path
    self.jobutils.check_ret_val = mock.MagicMock()
    return mock_res_setting_data