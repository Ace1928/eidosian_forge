from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_replicas as osc_share_replicas
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareReplicaSet(TestShareReplica):

    def setUp(self):
        super(TestShareReplicaSet, self).setUp()
        self.share_replica = manila_fakes.FakeShareReplica.create_one_replica()
        self.replicas_mock.get.return_value = self.share_replica
        self.cmd = osc_share_replicas.SetShareReplica(self.app, None)

    def test_share_replica_set_replica_state(self):
        new_replica_state = 'in_sync'
        arglist = [self.share_replica.id, '--replica-state', new_replica_state]
        verifylist = [('replica', self.share_replica.id), ('replica_state', new_replica_state)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.replicas_mock.reset_replica_state.assert_called_with(self.share_replica, new_replica_state)
        self.assertIsNone(result)

    def test_share_replica_set_replica_state_exception(self):
        new_replica_state = 'in_sync'
        arglist = [self.share_replica.id, '--replica-state', new_replica_state]
        verifylist = [('replica', self.share_replica.id), ('replica_state', new_replica_state)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.replicas_mock.reset_replica_state.side_effect = Exception()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_replica_set_status(self):
        new_status = 'available'
        arglist = [self.share_replica.id, '--status', new_status]
        verifylist = [('replica', self.share_replica.id), ('status', new_status)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.replicas_mock.reset_state.assert_called_with(self.share_replica, new_status)
        self.assertIsNone(result)

    def test_share_replica_set_status_exception(self):
        new_status = 'available'
        arglist = [self.share_replica.id, '--status', new_status]
        verifylist = [('replica', self.share_replica.id), ('status', new_status)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.replicas_mock.reset_state.side_effect = Exception()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_replica_set_nothing_defined(self):
        arglist = [self.share_replica.id]
        verifylist = [('replica', self.share_replica.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)