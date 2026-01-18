from unittest import mock
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils10
from os_win.utils import jobutils
@mock.patch.object(vmutils10.VMUtils10, 'get_vm_generation')
def _test_vm_has_s3_controller(self, vm_gen, mock_get_vm_gen):
    mock_get_vm_gen.return_value = vm_gen
    return self._vmutils._vm_has_s3_controller(mock.sentinel.fake_vm_name)