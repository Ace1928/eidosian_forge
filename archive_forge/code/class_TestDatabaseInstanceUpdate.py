from unittest import mock
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_instances
from troveclient.tests.osc.v1 import fakes
from troveclient.v1 import instances
class TestDatabaseInstanceUpdate(TestInstances):

    def setUp(self):
        super(TestDatabaseInstanceUpdate, self).setUp()
        self.cmd = database_instances.UpdateDatabaseInstance(self.app, None)

    @mock.patch.object(utils, 'find_resource')
    def test_instance_update(self, mock_find):
        args = ['instance1', '--name', 'new_instance_name', '--detach_replica_source', '--remove_configuration']
        mock_find.return_value = args[0]
        parsed_args = self.check_parser(self.cmd, args, [])
        result = self.cmd.take_action(parsed_args)
        self.instance_client.update.assert_called_with('instance1', None, 'new_instance_name', True, True, is_public=None, allowed_cidrs=None)
        self.assertIsNone(result)

    def test_instance_update_access(self):
        ins_id = '4c397f77-750d-43df-8fc5-f7388e4316ee'
        args = [ins_id, '--name', 'new_instance_name', '--is-private', '--allowed-cidr', '10.0.0.0/24', '--allowed-cidr', '10.0.1.0/24']
        parsed_args = self.check_parser(self.cmd, args, [])
        self.cmd.take_action(parsed_args)
        self.instance_client.update.assert_called_with(ins_id, None, 'new_instance_name', False, False, is_public=False, allowed_cidrs=['10.0.0.0/24', '10.0.1.0/24'])