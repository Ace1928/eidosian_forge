from unittest import mock
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_instances
from troveclient.tests.osc.v1 import fakes
from troveclient.v1 import instances
class TestDatabaseInstanceRestart(TestInstances):

    def setUp(self):
        super(TestDatabaseInstanceRestart, self).setUp()
        self.cmd = database_instances.RestartDatabaseInstance(self.app, None)

    @mock.patch.object(utils, 'find_resource')
    def test_instance_restart(self, mock_find):
        args = ['instance1']
        mock_find.return_value = args[0]
        parsed_args = self.check_parser(self.cmd, args, [])
        result = self.cmd.take_action(parsed_args)
        self.instance_client.restart.assert_called_with('instance1')
        self.assertIsNone(result)