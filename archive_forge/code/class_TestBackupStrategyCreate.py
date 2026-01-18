from troveclient.osc.v1 import database_backup_strategy
from troveclient.tests.osc.v1 import fakes
from troveclient.v1 import backup_strategy
class TestBackupStrategyCreate(TestBackupStrategy):

    def setUp(self):
        super(TestBackupStrategyCreate, self).setUp()
        self.cmd = database_backup_strategy.CreateDatabaseBackupStrategy(self.app, None)

    def test_create(self):
        args = ['--instance-id', 'fake_instance_id', '--swift-container', 'fake_container']
        parsed_args = self.check_parser(self.cmd, args, [])
        self.cmd.take_action(parsed_args)
        self.manager.create.assert_called_once_with(instance_id='fake_instance_id', swift_container='fake_container')