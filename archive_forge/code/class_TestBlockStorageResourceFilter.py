from unittest import mock
from cinderclient import api_versions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_resource_filter
class TestBlockStorageResourceFilter(volume_fakes.TestVolume):

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
        self.addCleanup(patcher.stop)
        self.supports_microversion_mock = patcher.start()
        self._set_mock_microversion(self.app.client_manager.volume.api_version.get_string())

    def _set_mock_microversion(self, mock_v):
        """Set a specific microversion for the mock supports_microversion()."""
        self.supports_microversion_mock.reset_mock(return_value=True)
        self.supports_microversion_mock.side_effect = lambda _, v: api_versions.APIVersion(v) <= api_versions.APIVersion(mock_v)