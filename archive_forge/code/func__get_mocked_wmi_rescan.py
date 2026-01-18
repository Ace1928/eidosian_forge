from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
def _get_mocked_wmi_rescan(self, return_value):
    conn = self._diskutils._conn_storage
    rescan_method = conn.Msft_StorageSetting.UpdateHostStorageCache
    rescan_method.return_value = return_value
    return rescan_method