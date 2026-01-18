from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient.osc.v1 import database_backups
from troveclient.tests.osc.v1 import fakes
class TestBackupShow(TestBackups):
    values = ('2015-05-16T14:22:28', 'mysql', '5.6', 'v-56', None, 'bk-1234', '1234', 'http://backup_srvr/database_backups/bk-1234.xbstream.gz.enc', 'bkp_1', None, '262db161-d3e4-4218-8bde-5bd879fc3e61', 0.11, 'COMPLETED', '2015-05-16T14:23:08')

    def setUp(self):
        super(TestBackupShow, self).setUp()
        self.cmd = database_backups.ShowDatabaseBackup(self.app, None)
        self.data = self.fake_backups.get_backup_bk_1234()
        self.backup_client.get.return_value = self.data
        self.columns = ('created', 'datastore', 'datastore_version', 'datastore_version_id', 'description', 'id', 'instance_id', 'locationRef', 'name', 'parent_id', 'project_id', 'size', 'status', 'updated')

    def test_show(self):
        args = ['bkp_1']
        parsed_args = self.check_parser(self.cmd, args, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.values, data)