import copy
import operator
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.osc.v2.networking_bgpvpn import bgpvpn
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
class TestShowBgpvpn(fakes.TestNeutronClientBgpvpn):

    def setUp(self):
        super(TestShowBgpvpn, self).setUp()
        self.cmd = bgpvpn.ShowBgpvpn(self.app, self.namespace)
        self.networkclient.find_bgpvpn = mock.Mock(side_effect=lambda name_or_id: {'id': name_or_id})

    def test_show_bgpvpn(self):
        fake_bgpvpn = fakes.create_one_bgpvpn()
        self.networkclient.get_bgpvpn = mock.Mock(return_value=fake_bgpvpn)
        arglist = [fake_bgpvpn['id']]
        verifylist = [('bgpvpn', fake_bgpvpn['id'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.networkclient.get_bgpvpn.assert_called_once_with(fake_bgpvpn['id'])
        self.assertEqual(sorted(sorted_columns), sorted(headers))