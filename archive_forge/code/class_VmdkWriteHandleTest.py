import ssl
from unittest import mock
import requests
from oslo_vmware import exceptions
from oslo_vmware import rw_handles
from oslo_vmware.tests import base
from oslo_vmware import vim_util
class VmdkWriteHandleTest(base.TestCase):
    """Tests for VmdkWriteHandle."""

    def setUp(self):
        super(VmdkWriteHandleTest, self).setUp()
        self._conn = mock.Mock()
        patcher = mock.patch('urllib3.connection.HTTPConnection')
        self.addCleanup(patcher.stop)
        HTTPConnectionMock = patcher.start()
        HTTPConnectionMock.return_value = self._conn

    def _create_mock_session(self, disk=True, progress=-1):
        device_url = mock.Mock()
        device_url.disk = disk
        device_url.url = 'http://*/ds/disk1.vmdk'
        lease_info = mock.Mock()
        lease_info.deviceUrl = [device_url]
        session = mock.Mock()

        def session_invoke_api_side_effect(module, method, *args, **kwargs):
            if module == session.vim:
                if method == 'ImportVApp':
                    return mock.Mock()
                elif method == 'HttpNfcLeaseProgress':
                    self.assertEqual(progress, kwargs['percent'])
                    return
            return lease_info
        session.invoke_api.side_effect = session_invoke_api_side_effect
        vim_cookie = mock.Mock()
        vim_cookie.name = 'name'
        vim_cookie.value = 'value'
        session.vim.client.cookiejar = [vim_cookie]
        return session

    def test_init_failure(self):
        session = self._create_mock_session(False)
        self.assertRaises(exceptions.VimException, rw_handles.VmdkWriteHandle, session, '10.1.2.3', 443, 'rp-1', 'folder-1', None, 100)

    def test_write(self):
        session = self._create_mock_session()
        handle = rw_handles.VmdkWriteHandle(session, '10.1.2.3', 443, 'rp-1', 'folder-1', None, 100)
        data = [1] * 10
        handle.write(data)
        self.assertEqual(len(data), handle._bytes_written)
        self._conn.putrequest.assert_called_once_with('PUT', '/ds/disk1.vmdk')
        self._conn.send.assert_called_once_with(data)

    def test_tell(self):
        session = self._create_mock_session()
        handle = rw_handles.VmdkWriteHandle(session, '10.1.2.3', 443, 'rp-1', 'folder-1', None, 100)
        data = [1] * 10
        handle.write(data)
        self.assertEqual(len(data), handle._bytes_written)
        self.assertEqual(len(data), handle.tell())

    def test_write_post(self):
        session = self._create_mock_session()
        handle = rw_handles.VmdkWriteHandle(session, '10.1.2.3', 443, 'rp-1', 'folder-1', None, 100, http_method='POST')
        data = [1] * 10
        handle.write(data)
        self.assertEqual(len(data), handle._bytes_written)
        self._conn.putrequest.assert_called_once_with('POST', '/ds/disk1.vmdk')
        self._conn.send.assert_called_once_with(data)

    def test_update_progress(self):
        vmdk_size = 100
        data_size = 10
        session = self._create_mock_session(True, 10)
        handle = rw_handles.VmdkWriteHandle(session, '10.1.2.3', 443, 'rp-1', 'folder-1', None, vmdk_size)
        handle.write([1] * data_size)
        handle.update_progress()

    def test_close(self):
        session = self._create_mock_session()
        handle = rw_handles.VmdkWriteHandle(session, '10.1.2.3', 443, 'rp-1', 'folder-1', None, 100)

        def session_invoke_api_side_effect(module, method, *args, **kwargs):
            if module == vim_util and method == 'get_object_property':
                return 'ready'
            self.assertEqual(session.vim, module)
            self.assertEqual('HttpNfcLeaseComplete', method)
        session.invoke_api = mock.Mock(side_effect=session_invoke_api_side_effect)
        handle._get_progress = mock.Mock(return_value=100)
        handle.close()
        self.assertEqual(2, session.invoke_api.call_count)

    def test_get_vm_incomplete_transfer(self):
        session = self._create_mock_session()
        handle = rw_handles.VmdkWriteHandle(session, '10.1.2.3', 443, 'rp-1', 'folder-1', None, 100)
        handle._get_progress = mock.Mock(return_value=99)
        session.invoke_api = mock.Mock()
        self.assertRaises(exceptions.ImageTransferException, handle.get_imported_vm)