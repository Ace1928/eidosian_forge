import ssl
from unittest import mock
import requests
from oslo_vmware import exceptions
from oslo_vmware import rw_handles
from oslo_vmware.tests import base
from oslo_vmware import vim_util
class VmdkHandleTest(base.TestCase):
    """Tests for VmdkHandle."""

    def test_find_vmdk_url(self):
        device_url_0 = mock.Mock()
        device_url_0.disk = False
        device_url_1 = mock.Mock()
        device_url_1.disk = True
        device_url_1.url = 'https://*/ds1/vm1.vmdk'
        device_url_1.sslThumbprint = '11:22:33:44:55'
        lease_info = mock.Mock()
        lease_info.deviceUrl = [device_url_0, device_url_1]
        host = '10.1.2.3'
        port = 443
        exp_url = 'https://%s:%d/ds1/vm1.vmdk' % (host, port)
        vmw_http_file = rw_handles.VmdkHandle(None, None, None, None)
        url, thumbprint = vmw_http_file._find_vmdk_url(lease_info, host, port)
        self.assertEqual(exp_url, url)
        self.assertEqual('11:22:33:44:55', thumbprint)

    def test_update_progress(self):
        session = mock.Mock()
        lease = mock.Mock()
        handle = rw_handles.VmdkHandle(session, lease, 'fake-url', None)
        handle._get_progress = mock.Mock(return_value=50)
        handle.update_progress()
        session.invoke_api.assert_called_once_with(session.vim, 'HttpNfcLeaseProgress', lease, percent=50)

    def test_update_progress_with_error(self):
        session = mock.Mock()
        handle = rw_handles.VmdkHandle(session, None, 'fake-url', None)
        handle._get_progress = mock.Mock(return_value=0)
        session.invoke_api.side_effect = exceptions.VimException(None)
        self.assertRaises(exceptions.VimException, handle.update_progress)

    def test_fileno(self):
        session = mock.Mock()
        handle = rw_handles.VmdkHandle(session, None, 'fake-url', None)
        self.assertRaises(IOError, handle.fileno)

    def test_release_lease_incomplete_transfer(self):
        session = mock.Mock()
        handle = rw_handles.VmdkHandle(session, None, 'fake-url', None)
        handle._get_progress = mock.Mock(return_value=99)
        session.invoke_api = mock.Mock()
        handle._release_lease()
        session.invoke_api.assert_called_with(handle._session.vim, 'HttpNfcLeaseAbort', handle._lease)