from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient.osc.v1 import database_backups
from troveclient.tests.osc.v1 import fakes
class TestBackups(fakes.TestDatabasev1):
    fake_backups = fakes.FakeBackups()

    def setUp(self):
        super(TestBackups, self).setUp()
        self.mock_client = self.app.client_manager.database
        self.backup_client = self.app.client_manager.database.backups
        self.instance_client = self.app.client_manager.database.instances