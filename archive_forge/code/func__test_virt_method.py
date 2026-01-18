from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import jobutils
def _test_virt_method(self, vsms_method_name, return_count, utils_method_name, with_mock_vm, *args, **kwargs):
    mock_svc = mock.MagicMock()
    self.jobutils._vs_man_svc_attr = mock_svc
    vsms_method = getattr(mock_svc, vsms_method_name)
    mock_rsd = self._mock_vsms_method(vsms_method, return_count)
    if with_mock_vm:
        mock_vm = mock.MagicMock()
        mock_vm.path_.return_value = mock.sentinel.vm_path
        getattr(self.jobutils, utils_method_name)(mock_rsd, mock_vm)
    else:
        getattr(self.jobutils, utils_method_name)(mock_rsd)
    if args:
        vsms_method.assert_called_once_with(*args)
    else:
        vsms_method.assert_called_once_with(**kwargs)