import re
from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils10
def _test_is_host_guarded(self, return_code=0, is_host_guarded=True):
    hgs_config = self._hostutils._conn_hgs.MSFT_HgsClientConfiguration
    hgs_config.Get.return_value = (return_code, mock.MagicMock(IsHostGuarded=is_host_guarded))
    expected_result = is_host_guarded and (not return_code)
    result = self._hostutils.is_host_guarded()
    self.assertEqual(expected_result, result)