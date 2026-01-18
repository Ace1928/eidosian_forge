import copy
from unittest import mock
from oslo_i18n import fixture as i18n_fixture
import suds
from oslo_vmware import exceptions
from oslo_vmware.tests import base
from oslo_vmware import vim
class VimTest(base.TestCase):
    """Test class for Vim."""

    def setUp(self):
        super(VimTest, self).setUp()
        patcher = mock.patch('oslo_vmware.service.CompatibilitySudsClient')
        self.addCleanup(patcher.stop)
        self.SudsClientMock = patcher.start()
        self.useFixture(i18n_fixture.ToggleLazy(True))

    @mock.patch.object(vim.Vim, '__getattr__', autospec=True)
    def test_service_content(self, getattr_mock):
        getattr_ret = mock.Mock()
        getattr_mock.side_effect = lambda *args: getattr_ret
        vim_obj = vim.Vim()
        vim_obj.service_content
        getattr_mock.assert_called_once_with(vim_obj, 'RetrieveServiceContent')
        getattr_ret.assert_called_once_with('ServiceInstance')
        self.assertEqual(self.SudsClientMock.return_value, vim_obj.client)
        self.assertEqual(getattr_ret.return_value, vim_obj.service_content)

    def test_configure_non_default_host_port(self):
        vim_obj = vim.Vim('https', 'www.test.com', 12345)
        self.assertEqual('https://www.test.com:12345/sdk/vimService.wsdl', vim_obj.wsdl_url)
        self.assertEqual('https://www.test.com:12345/sdk', vim_obj.soap_url)

    def test_configure_ipv6(self):
        vim_obj = vim.Vim('https', '::1')
        self.assertEqual('https://[::1]/sdk/vimService.wsdl', vim_obj.wsdl_url)
        self.assertEqual('https://[::1]/sdk', vim_obj.soap_url)

    def test_configure_ipv6_and_non_default_host_port(self):
        vim_obj = vim.Vim('https', '::1', 12345)
        self.assertEqual('https://[::1]:12345/sdk/vimService.wsdl', vim_obj.wsdl_url)
        self.assertEqual('https://[::1]:12345/sdk', vim_obj.soap_url)

    def test_configure_with_wsdl_url_override(self):
        vim_obj = vim.Vim('https', 'www.example.com', wsdl_url='https://test.com/sdk/vimService.wsdl')
        self.assertEqual('https://test.com/sdk/vimService.wsdl', vim_obj.wsdl_url)
        self.assertEqual('https://www.example.com/sdk', vim_obj.soap_url)