from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def _prepare_mock_disk(self):
    mock_disk = mock.MagicMock()
    mock_disk.HostResource = [self._FAKE_HOST_RESOURCE]
    mock_disk.path.return_value.RelPath = self._FAKE_RES_PATH
    mock_disk.ResourceSubType = self._vmutils._HARD_DISK_RES_SUB_TYPE
    self._vmutils._conn.query.return_value = [mock_disk]
    return mock_disk