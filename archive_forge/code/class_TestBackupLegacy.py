from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_backup
class TestBackupLegacy(volume_fakes.TestVolume):

    def setUp(self):
        super().setUp()
        self.backups_mock = self.volume_client.backups
        self.backups_mock.reset_mock()
        self.volumes_mock = self.volume_client.volumes
        self.volumes_mock.reset_mock()
        self.snapshots_mock = self.volume_client.volume_snapshots
        self.snapshots_mock.reset_mock()
        self.restores_mock = self.volume_client.restores
        self.restores_mock.reset_mock()