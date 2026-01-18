from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_replicas as osc_share_replicas
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareReplicaDelete(TestShareReplica):

    def setUp(self):
        super(TestShareReplicaDelete, self).setUp()
        self.share_replica = manila_fakes.FakeShareReplica.create_one_replica()
        self.replicas_mock.get.return_value = self.share_replica
        self.cmd = osc_share_replicas.DeleteShareReplica(self.app, None)

    def test_share_replica_delete_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_replica_delete(self):
        arglist = [self.share_replica.id]
        verifylist = [('replica', [self.share_replica.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.replicas_mock.delete.assert_called_with(self.share_replica, force=False)
        self.assertIsNone(result)

    def test_share_replica_delete_force(self):
        arglist = [self.share_replica.id, '--force']
        verifylist = [('replica', [self.share_replica.id]), ('force', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.replicas_mock.delete.assert_called_with(self.share_replica, force=True)
        self.assertIsNone(result)

    def test_share_replica_delete_multiple(self):
        share_replicas = manila_fakes.FakeShareReplica.create_share_replicas(count=2)
        arglist = [share_replicas[0].id, share_replicas[1].id]
        verifylist = [('replica', [share_replicas[0].id, share_replicas[1].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertEqual(self.replicas_mock.delete.call_count, len(share_replicas))
        self.assertIsNone(result)

    def test_share_snapshot_delete_exception(self):
        arglist = [self.share_replica.id]
        verifylist = [('replica', [self.share_replica.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.replicas_mock.delete.side_effect = exceptions.CommandError()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_replica_delete_wait(self):
        arglist = [self.share_replica.id, '--wait']
        verifylist = [('replica', [self.share_replica.id]), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_delete', return_value=True):
            result = self.cmd.take_action(parsed_args)
            self.replicas_mock.delete.assert_called_with(self.share_replica, force=False)
            self.replicas_mock.get.assert_called_with(self.share_replica.id)
            self.assertIsNone(result)

    def test_share_replica_delete_wait_exception(self):
        arglist = [self.share_replica.id, '--wait']
        verifylist = [('replica', [self.share_replica.id]), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_delete', return_value=False):
            self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)