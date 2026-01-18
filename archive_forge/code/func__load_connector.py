import os
from unittest import mock
import ddt
from os_brick.initiator.windows import smbfs
from os_brick.remotefs import windows_remotefs
from os_brick.tests.windows import test_base
@mock.patch.object(windows_remotefs, 'WindowsRemoteFsClient')
def _load_connector(self, mock_remotefs_cls, *args, **kwargs):
    self._connector = smbfs.WindowsSMBFSConnector(*args, **kwargs)
    self._remotefs = mock_remotefs_cls.return_value
    self._vhdutils = self._connector._vhdutils
    self._diskutils = self._connector._diskutils