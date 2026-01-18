from unittest import mock
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_instances
from troveclient.tests.osc.v1 import fakes
from troveclient.v1 import instances
class TestDatabaseInstanceEjectReplicaSource(TestInstances):

    def setUp(self):
        super(TestDatabaseInstanceEjectReplicaSource, self).setUp()
        self.cmd = database_instances.EjectDatabaseInstanceReplicaSource(self.app, None)

    @mock.patch.object(utils, 'find_resource')
    def test_eject_replica_source(self, mock_find):
        args = ['instance']
        mock_find.return_value = args[0]
        parsed_args = self.check_parser(self.cmd, args, [])
        result = self.cmd.take_action(parsed_args)
        self.instance_client.eject_replica_source.assert_called_with('instance')
        self.assertIsNone(result)