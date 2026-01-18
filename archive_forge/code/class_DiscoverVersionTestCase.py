from unittest import mock
from zunclient import api_versions
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import versions
class DiscoverVersionTestCase(utils.TestCase):

    def setUp(self):
        super(DiscoverVersionTestCase, self).setUp()
        self.orig_max = api_versions.MAX_API_VERSION
        self.orig_min = api_versions.MIN_API_VERSION
        self.addCleanup(self._clear_fake_version)

    def _clear_fake_version(self):
        api_versions.MAX_API_VERSION = self.orig_max
        api_versions.MIN_API_VERSION = self.orig_min

    def test_server_is_too_new(self):
        fake_client = mock.MagicMock()
        fake_client.versions.get_current.return_value = mock.MagicMock(max_version='1.7', min_version='1.4')
        api_versions.MAX_API_VERSION = '1.3'
        api_versions.MIN_API_VERSION = '1.1'
        self.assertRaises(exceptions.UnsupportedVersion, api_versions.discover_version, fake_client, api_versions.APIVersion('1.latest'))

    def test_server_is_too_old(self):
        fake_client = mock.MagicMock()
        fake_client.versions.get_current.return_value = mock.MagicMock(max_version='1.7', min_version='1.4')
        api_versions.MAX_API_VERSION = '1.10'
        api_versions.MIN_API_VERSION = '1.9'
        self.assertRaises(exceptions.UnsupportedVersion, api_versions.discover_version, fake_client, api_versions.APIVersion('1.latest'))

    def test_server_end_version_is_the_latest_one(self):
        fake_client = mock.MagicMock()
        fake_client.versions.get_current.return_value = mock.MagicMock(max_version='1.7', min_version='1.4')
        api_versions.MAX_API_VERSION = '1.11'
        api_versions.MIN_API_VERSION = '1.1'
        self.assertEqual('1.7', api_versions.discover_version(fake_client, api_versions.APIVersion('1.latest')).get_string())

    def test_client_end_version_is_the_latest_one(self):
        fake_client = mock.MagicMock()
        fake_client.versions.get_current.return_value = mock.MagicMock(max_version='1.16', min_version='1.4')
        api_versions.MAX_API_VERSION = '1.11'
        api_versions.MIN_API_VERSION = '1.1'
        self.assertEqual('1.11', api_versions.discover_version(fake_client, api_versions.APIVersion('1.latest')).get_string())

    def test_server_without_microversion(self):
        fake_client = mock.MagicMock()
        fake_client.versions.get_current.return_value = mock.MagicMock(max_version='', min_version='')
        api_versions.MAX_API_VERSION = '1.11'
        api_versions.MIN_API_VERSION = '1.1'
        self.assertEqual('1.1', api_versions.discover_version(fake_client, api_versions.APIVersion('1.latest')).get_string())

    def test_server_without_microversion_and_no_version_field(self):
        fake_client = mock.MagicMock()
        fake_client.versions.get_current.return_value = versions.Version(None, {})
        api_versions.MAX_API_VERSION = '1.11'
        api_versions.MIN_API_VERSION = '1.1'
        self.assertEqual('1.1', api_versions.discover_version(fake_client, api_versions.APIVersion('1.latest')).get_string())