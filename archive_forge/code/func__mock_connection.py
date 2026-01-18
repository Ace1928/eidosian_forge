import ssl
from unittest import mock
import requests
from oslo_vmware import exceptions
from oslo_vmware import rw_handles
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def _mock_connection(self, read_data='fake-data'):
    self._resp = mock.Mock()
    self._resp.read.return_value = read_data
    self._conn = mock.Mock()
    self._conn.getresponse.return_value = self._resp
    patcher = mock.patch('urllib3.connection.HTTPConnection')
    self.addCleanup(patcher.stop)
    HTTPConnectionMock = patcher.start()
    HTTPConnectionMock.return_value = self._conn