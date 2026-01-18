from unittest import mock
from osc_lib import utils
from troveclient.osc.v1 import database_logs
from troveclient.tests.osc.v1 import fakes
class TestLogs(fakes.TestDatabasev1):
    fake_logs = fakes.FakeLogs()

    def setUp(self):
        super(TestLogs, self).setUp()
        self.instance_client = self.app.client_manager.database.instances