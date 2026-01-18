import operator
from unittest import mock
from osc_lib.tests.utils import ParserException
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
class TestShowRouterAssoc(fakes.TestNeutronClientBgpvpn):

    def setUp(self):
        super(TestShowRouterAssoc, self).setUp()
        self.cmd = fakes.ShowBgpvpnFakeRouterAssoc(self.app, self.namespace)
        self.networkclient.find_bgpvpn = mock.Mock(side_effect=lambda name_or_id: {'id': name_or_id})

    def test_show_router_association(self):
        fake_bgpvpn = fakes.create_one_bgpvpn()
        fake_res = fakes.create_one_resource()
        fake_res_assoc = fakes.create_one_resource_association(fake_res, {'advertise_extra_routes': True})
        self.networkclient.get_bgpvpn_router_association = mock.Mock(return_value=fake_res_assoc)
        arglist = [fake_res_assoc['id'], fake_bgpvpn['id']]
        verifylist = [('resource_association_id', fake_res_assoc['id']), ('bgpvpn', fake_bgpvpn['id'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        cols, data = self.cmd.take_action(parsed_args)
        self.networkclient.get_bgpvpn_router_association.assert_called_once_with(fake_bgpvpn['id'], fake_res_assoc['id'])
        self.assertEqual(sorted_columns, cols)
        self.assertEqual(data, _get_data(fake_res_assoc))