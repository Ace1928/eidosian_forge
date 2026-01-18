from unittest import mock
import novaclient
from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient import utils as nutils
from novaclient.v2 import versions
class CheckHeadersTestCase(utils.TestCase):

    def setUp(self):
        super(CheckHeadersTestCase, self).setUp()
        mock_log_patch = mock.patch('novaclient.api_versions.LOG')
        self.mock_log = mock_log_patch.start()
        self.addCleanup(mock_log_patch.stop)

    def test_legacy_microversion_is_specified(self):
        response = mock.MagicMock(headers={api_versions.LEGACY_HEADER_NAME: ''})
        api_versions.check_headers(response, api_versions.APIVersion('2.2'))
        self.assertFalse(self.mock_log.warning.called)
        response = mock.MagicMock(headers={})
        api_versions.check_headers(response, api_versions.APIVersion('2.2'))
        self.assertTrue(self.mock_log.warning.called)

    def test_generic_microversion_is_specified(self):
        response = mock.MagicMock(headers={api_versions.HEADER_NAME: ''})
        api_versions.check_headers(response, api_versions.APIVersion('2.27'))
        self.assertFalse(self.mock_log.warning.called)
        response = mock.MagicMock(headers={})
        api_versions.check_headers(response, api_versions.APIVersion('2.27'))
        self.assertTrue(self.mock_log.warning.called)

    def test_microversion_is_not_specified(self):
        response = mock.MagicMock(headers={api_versions.LEGACY_HEADER_NAME: ''})
        api_versions.check_headers(response, api_versions.APIVersion('2.2'))
        self.assertFalse(self.mock_log.warning.called)
        response = mock.MagicMock(headers={})
        api_versions.check_headers(response, api_versions.APIVersion('2.0'))
        self.assertFalse(self.mock_log.warning.called)