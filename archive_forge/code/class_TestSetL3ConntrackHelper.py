from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import l3_conntrack_helper
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestSetL3ConntrackHelper(TestConntrackHelper):

    def setUp(self):
        super(TestSetL3ConntrackHelper, self).setUp()
        attrs = {'router_id': self.router.id}
        self.ct_helper = network_fakes.FakeL3ConntrackHelper.create_one_l3_conntrack_helper(attrs)
        self.network_client.update_conntrack_helper = mock.Mock(return_value=None)
        self.cmd = l3_conntrack_helper.SetConntrackHelper(self.app, self.namespace)

    def test_set_nothing(self):
        arglist = [self.router.id, self.ct_helper.id]
        verifylist = [('router', self.router.id), ('conntrack_helper_id', self.ct_helper.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.update_conntrack_helper.assert_called_once_with(self.ct_helper.id, self.router.id)
        self.assertIsNone(result)

    def test_set_port(self):
        arglist = [self.router.id, self.ct_helper.id, '--port', '124']
        verifylist = [('router', self.router.id), ('conntrack_helper_id', self.ct_helper.id), ('port', 124)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.update_conntrack_helper.assert_called_once_with(self.ct_helper.id, self.router.id, port=124)
        self.assertIsNone(result)