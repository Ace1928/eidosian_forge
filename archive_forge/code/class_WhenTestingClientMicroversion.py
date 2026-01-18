from unittest import mock
from keystoneauth1 import session
from requests_mock.contrib import fixture
import testtools
from barbicanclient import client
from barbicanclient import exceptions
from barbicanclient.exceptions import UnsupportedVersion
from barbicanclient.tests.utils import get_server_supported_versions
from barbicanclient.tests.utils import get_version_endpoint
from barbicanclient.tests.utils import mock_session
from barbicanclient.tests.utils import mock_session_get
from barbicanclient.tests.utils import mock_session_get_endpoint
class WhenTestingClientMicroversion(TestClient):

    def _create_mock_session(self, requested_version, server_max_version, server_min_version, endpoint):
        sess = mock_session()
        mock_session_get_endpoint(sess, get_version_endpoint(endpoint))
        mock_session_get(sess, get_server_supported_versions(server_min_version, server_max_version))
        return sess

    def _test_client_creation_with_endpoint(self, requested_version, server_max_version, server_min_version, endpoint):
        sess = self._create_mock_session(requested_version, server_max_version, server_min_version, endpoint)
        client.Client(session=sess, microversion=requested_version)
        headers = {'Accept': 'application/json', 'OpenStack-API-Version': 'key-manager 1.1'}
        sess.get.assert_called_with(get_version_endpoint(endpoint), headers=headers, authenticated=None)

    def _mock_session_and_get_client(self, requested_version, server_max_version, server_min_version, endpoint=None):
        sess = self._create_mock_session(requested_version, server_max_version, server_min_version, endpoint)
        return client.Client(session=sess, microversion=requested_version)

    def test_fails_when_requesting_invalid_microversion(self):
        self.assertRaises(TypeError, client.Client, session=self.session, endpoint=self.endpoint, project_id=self.project_id, microversion='a')

    def test_fails_when_requesting_unsupported_microversion(self):
        self.assertRaises(UnsupportedVersion, client.Client, session=self.session, endpoint=self.endpoint, project_id=self.project_id, microversion='1.9')

    def test_fails_when_requesting_unsupported_version(self):
        self.assertRaises(UnsupportedVersion, client.Client, session=self.session, endpoint=self.endpoint, project_id=self.project_id, version='v0')

    def test_passes_without_providing_endpoint(self):
        requested_version = None
        server_max_version = (1, 1)
        server_min_version = (1, 0)
        endpoint = None
        self._test_client_creation_with_endpoint(requested_version, server_max_version, server_min_version, endpoint)

    def test_passes_with_custom_endpoint(self):
        requested_version = None
        server_max_version = (1, 1)
        server_min_version = (1, 0)
        endpoint = self.endpoint
        self._test_client_creation_with_endpoint(requested_version, server_max_version, server_min_version, endpoint)

    def test_passes_with_default_microversion_as_1_1(self):
        requested_version = None
        server_max_version = (1, 1)
        server_min_version = (1, 0)
        c = self._mock_session_and_get_client(requested_version, server_max_version, server_min_version)
        self.assertEqual('1.1', c.client.microversion)

    def test_passes_with_default_microversion_as_1_0(self):
        requested_version = None
        server_max_version = (1, 0)
        server_min_version = (1, 0)
        c = self._mock_session_and_get_client(requested_version, server_max_version, server_min_version)
        self.assertEqual('1.0', c.client.microversion)

    def test_fails_requesting_higher_microversion_than_supported_by_server(self):
        requested_version = '1.1'
        server_max_version = (1, 0)
        server_min_version = (1, 0)
        sess = self._create_mock_session(requested_version, server_max_version, server_min_version, self.endpoint)
        self.assertRaises(UnsupportedVersion, client.Client, session=sess, endpoint=self.endpoint, microversion=requested_version)

    def test_fails_requesting_lower_microversion_than_supported_by_server(self):
        requested_version = '1.0'
        server_max_version = (1, 1)
        server_min_version = (1, 1)
        sess = self._create_mock_session(requested_version, server_max_version, server_min_version, self.endpoint)
        self.assertRaises(UnsupportedVersion, client.Client, session=sess, endpoint=self.endpoint, microversion=requested_version)

    def test_passes_with_stable_server_version(self):
        requested_version = '1.0'
        server_max_version = None
        server_min_version = None
        c = self._mock_session_and_get_client(requested_version, server_max_version, server_min_version)
        self.assertEqual(requested_version, c.client.microversion)