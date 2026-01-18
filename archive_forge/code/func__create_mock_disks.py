from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def _create_mock_disks(self):
    mock_rasd1 = mock.MagicMock()
    mock_rasd1.ResourceSubType = self._vmutils._HARD_DISK_RES_SUB_TYPE
    mock_rasd1.HostResource = [self._FAKE_VHD_PATH]
    mock_rasd1.Connection = [self._FAKE_VHD_PATH]
    mock_rasd1.Parent = self._FAKE_CTRL_PATH
    mock_rasd1.Address = self._FAKE_ADDRESS
    mock_rasd1.HostResource = [self._FAKE_VHD_PATH]
    mock_rasd2 = mock.MagicMock()
    mock_rasd2.ResourceSubType = self._vmutils._PHYS_DISK_RES_SUB_TYPE
    mock_rasd2.HostResource = [self._FAKE_VOLUME_DRIVE_PATH]
    return [mock_rasd1, mock_rasd2]