from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import l3_conntrack_helper
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestListL3ConntrackHelper(TestConntrackHelper):

    def setUp(self):
        super(TestListL3ConntrackHelper, self).setUp()
        attrs = {'router_id': self.router.id}
        ct_helpers = network_fakes.FakeL3ConntrackHelper.create_l3_conntrack_helpers(attrs, count=3)
        self.columns = ('ID', 'Router ID', 'Helper', 'Protocol', 'Port')
        self.data = []
        for ct_helper in ct_helpers:
            self.data.append((ct_helper.id, ct_helper.router_id, ct_helper.helper, ct_helper.protocol, ct_helper.port))
        self.network_client.conntrack_helpers = mock.Mock(return_value=ct_helpers)
        self.cmd = l3_conntrack_helper.ListConntrackHelper(self.app, self.namespace)

    def test_conntrack_helpers_list(self):
        arglist = [self.router.id]
        verifylist = [('router', self.router.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.conntrack_helpers.assert_called_once_with(self.router.id)
        self.assertEqual(self.columns, columns)
        list_data = list(data)
        self.assertEqual(len(self.data), len(list_data))
        for index in range(len(list_data)):
            self.assertEqual(self.data[index], list_data[index])