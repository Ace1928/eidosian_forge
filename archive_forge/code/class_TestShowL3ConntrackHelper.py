from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import l3_conntrack_helper
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestShowL3ConntrackHelper(TestConntrackHelper):

    def setUp(self):
        super(TestShowL3ConntrackHelper, self).setUp()
        attrs = {'router_id': self.router.id}
        self.ct_helper = network_fakes.FakeL3ConntrackHelper.create_one_l3_conntrack_helper(attrs)
        self.columns = ('helper', 'id', 'port', 'protocol', 'router_id')
        self.data = (self.ct_helper.helper, self.ct_helper.id, self.ct_helper.port, self.ct_helper.protocol, self.ct_helper.router_id)
        self.network_client.get_conntrack_helper = mock.Mock(return_value=self.ct_helper)
        self.cmd = l3_conntrack_helper.ShowConntrackHelper(self.app, self.namespace)

    def test_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_show_default_options(self):
        arglist = [self.router.id, self.ct_helper.id]
        verifylist = [('router', self.router.id), ('conntrack_helper_id', self.ct_helper.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.get_conntrack_helper.assert_called_once_with(self.ct_helper.id, self.router.id)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)