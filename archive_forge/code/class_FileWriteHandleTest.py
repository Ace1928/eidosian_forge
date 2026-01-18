import ssl
from unittest import mock
import requests
from oslo_vmware import exceptions
from oslo_vmware import rw_handles
from oslo_vmware.tests import base
from oslo_vmware import vim_util
class FileWriteHandleTest(base.TestCase):
    """Tests for FileWriteHandle."""

    def setUp(self):
        super(FileWriteHandleTest, self).setUp()
        vim_cookie = mock.Mock()
        vim_cookie.name = 'name'
        vim_cookie.value = 'value'
        self._conn = mock.Mock()
        patcher = mock.patch('urllib3.connection.HTTPConnection')
        self.addCleanup(patcher.stop)
        HTTPConnectionMock = patcher.start()
        HTTPConnectionMock.return_value = self._conn
        self.vmw_http_write_file = rw_handles.FileWriteHandle('10.1.2.3', 443, 'dc-0', 'ds-0', [vim_cookie], '1.vmdk', 100, 'http')

    def test_write(self):
        self.vmw_http_write_file.write(None)
        self._conn.send.assert_called_once_with(None)

    def test_close(self):
        self.vmw_http_write_file.close()
        self._conn.getresponse.assert_called_once_with()
        self._conn.close.assert_called_once_with()