from unittest import mock
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_instances
from troveclient.tests.osc.v1 import fakes
from troveclient.v1 import instances
class TestInstanceReplicaDetach(TestInstances):

    def setUp(self):
        super(TestInstanceReplicaDetach, self).setUp()
        self.cmd = database_instances.DetachDatabaseInstanceReplica(self.app, None)

    @mock.patch.object(utils, 'find_resource')
    def test_instance_replica_detach(self, mock_find):
        args = ['instance']
        mock_find.return_value = args[0]
        parsed_args = self.check_parser(self.cmd, args, [])
        result = self.cmd.take_action(parsed_args)
        self.instance_client.update.assert_called_with('instance', detach_replica_source=True)
        self.assertIsNone(result)