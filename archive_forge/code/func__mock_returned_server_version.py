from unittest import mock
import ddt
from cinderclient import api_versions
from cinderclient import exceptions
from cinderclient.tests.unit import test_utils
from cinderclient.tests.unit import utils
from cinderclient.v3 import client
def _mock_returned_server_version(self, server_version, server_min_version):
    version_mock = mock.MagicMock(version=server_version, min_version=server_min_version, status='CURRENT')
    val = [version_mock]
    if not server_version and (not server_min_version):
        val = []
    self.fake_client.services.server_api_version.return_value = val