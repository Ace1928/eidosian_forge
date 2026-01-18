from troveclient.osc.v1 import database_backup_strategy
from troveclient.tests.osc.v1 import fakes
from troveclient.v1 import backup_strategy
class TestBackupStrategy(fakes.TestDatabasev1):

    def setUp(self):
        super(TestBackupStrategy, self).setUp()
        self.manager = self.app.client_manager.database.backup_strategies