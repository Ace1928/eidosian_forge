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
def _test_client_creation_with_endpoint(self, requested_version, server_max_version, server_min_version, endpoint):
    sess = self._create_mock_session(requested_version, server_max_version, server_min_version, endpoint)
    client.Client(session=sess, microversion=requested_version)
    headers = {'Accept': 'application/json', 'OpenStack-API-Version': 'key-manager 1.1'}
    sess.get.assert_called_with(get_version_endpoint(endpoint), headers=headers, authenticated=None)