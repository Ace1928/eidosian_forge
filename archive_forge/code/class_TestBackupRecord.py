from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import backup_record
class TestBackupRecord(volume_fakes.TestVolume):

    def setUp(self):
        super().setUp()
        self.backups_mock = self.volume_client.backups
        self.backups_mock.reset_mock()