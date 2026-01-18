from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_replicas as osc_share_replicas
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareReplicaList(TestShareReplica):
    columns = ['id', 'status', 'replica_state', 'share_id', 'host', 'availability_zone', 'updated_at']
    column_headers = utils.format_column_headers(columns)

    def setUp(self):
        super(TestShareReplicaList, self).setUp()
        self.share = manila_fakes.FakeShare.create_one_share()
        self.shares_mock.get.return_value = self.share
        self.replicas_list = manila_fakes.FakeShareReplica.create_share_replicas(count=2)
        self.replicas_mock.list.return_value = self.replicas_list
        self.values = (oscutils.get_dict_properties(i._info, self.columns) for i in self.replicas_list)
        self.cmd = osc_share_replicas.ListShareReplica(self.app, None)

    def test_share_replica_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.replicas_mock.list.assert_called_with(share=None)
        self.assertEqual(self.column_headers, columns)
        self.assertEqual(list(self.values), list(data))

    def test_share_replica_list_for_share(self):
        arglist = ['--share', self.share.id]
        verifylist = [('share', self.share.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.replicas_mock.list.assert_called_with(share=self.share)
        self.assertEqual(self.column_headers, columns)
        self.assertEqual(list(self.values), list(data))