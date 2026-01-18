from unittest import mock
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_instances
from troveclient.tests.osc.v1 import fakes
from troveclient.v1 import instances
class TestDatabaseInstanceForceDelete(TestInstances):

    def setUp(self):
        super(TestDatabaseInstanceForceDelete, self).setUp()
        self.cmd = database_instances.ForceDeleteDatabaseInstance(self.app, None)

    @mock.patch.object(utils, 'find_resource')
    def test_instance_force_delete(self, mock_find):
        args = ['instance1']
        mock_find.return_value = args[0]
        parsed_args = self.check_parser(self.cmd, args, [])
        result = self.cmd.take_action(parsed_args)
        self.instance_client.reset_status.assert_called_with('instance1')
        self.instance_client.delete.assert_called_with('instance1')
        self.assertIsNone(result)

    @mock.patch.object(utils, 'find_resource')
    def test_instance_force_delete_with_exception(self, mock_find):
        args = ['fakeinstance']
        parsed_args = self.check_parser(self.cmd, args, [])
        mock_find.return_value = args[0]
        self.instance_client.delete.side_effect = exceptions.CommandError
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)