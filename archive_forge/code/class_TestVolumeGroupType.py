from unittest import mock
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group_type
class TestVolumeGroupType(volume_fakes.TestVolume):

    def setUp(self):
        super().setUp()
        self.volume_group_types_mock = self.volume_client.group_types
        self.volume_group_types_mock.reset_mock()