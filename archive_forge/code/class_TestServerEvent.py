from unittest import mock
import iso8601
from novaclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import server_event
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
class TestServerEvent(compute_fakes.TestComputev2):
    fake_server = compute_fakes.create_one_server()

    def setUp(self):
        super(TestServerEvent, self).setUp()
        patcher = mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
        self.addCleanup(patcher.stop)
        self.supports_microversion_mock = patcher.start()
        self._set_mock_microversion(self.compute_client.api_version.get_string())

    def _set_mock_microversion(self, mock_v):
        """Set a specific microversion for the mock supports_microversion()."""
        self.supports_microversion_mock.reset_mock(return_value=True)
        self.supports_microversion_mock.side_effect = lambda _, v: api_versions.APIVersion(v) <= api_versions.APIVersion(mock_v)