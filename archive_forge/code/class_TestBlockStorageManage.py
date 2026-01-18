from unittest import mock
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v2 import fakes as v2_volume_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_manage
class TestBlockStorageManage(v2_volume_fakes.TestVolume):

    def setUp(self):
        super().setUp()
        self.volumes_mock = self.volume_client.volumes
        self.volumes_mock.reset_mock()
        self.snapshots_mock = self.volume_client.volume_snapshots
        self.snapshots_mock.reset_mock()