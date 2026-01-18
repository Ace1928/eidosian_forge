from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def _prepare_get_vm_controller(self, resource_sub_type, mock_get_element_associated_class):
    self._lookup_vm()
    mock_rasds = mock.MagicMock()
    mock_rasds.path_.return_value = self._FAKE_RES_PATH
    mock_rasds.ResourceSubType = resource_sub_type
    mock_rasds.Address = self._FAKE_ADDRESS
    mock_get_element_associated_class.return_value = [mock_rasds]