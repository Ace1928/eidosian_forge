from troveclient.osc.v1 import database_backup_strategy
from troveclient.tests.osc.v1 import fakes
from troveclient.v1 import backup_strategy
class TestBackupStrategyList(TestBackupStrategy):

    def setUp(self):
        super(TestBackupStrategyList, self).setUp()
        self.cmd = database_backup_strategy.ListDatabaseBackupStrategies(self.app, None)

    def test_list(self):
        item = backup_strategy.BackupStrategy(None, {'project_id': 'fake_project_id', 'instance_id': 'fake_instance_id', 'swift_container': 'fake_container'})
        self.manager.list.return_value = [item]
        parsed_args = self.check_parser(self.cmd, [], [])
        columns, data = self.cmd.take_action(parsed_args)
        self.manager.list.assert_called_once_with(instance_id=None, project_id=None)
        self.assertEqual(database_backup_strategy.ListDatabaseBackupStrategies.columns, columns)
        self.assertEqual([('fake_project_id', 'fake_instance_id', 'fake_container')], data)