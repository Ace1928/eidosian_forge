import copy
import operator
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.osc.v2.networking_bgpvpn import bgpvpn
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
class TestCreateBgpvpn(fakes.TestNeutronClientBgpvpn):

    def setUp(self):
        super(TestCreateBgpvpn, self).setUp()
        self.cmd = bgpvpn.CreateBgpvpn(self.app, self.namespace)

    def test_create_bgpvpn_with_no_args(self):
        fake_bgpvpn = fakes.create_one_bgpvpn()
        self.networkclient.create_bgpvpn = mock.Mock(return_value=fake_bgpvpn)
        arglist = []
        verifylist = [('project', None), ('name', None), ('type', 'l3'), ('vni', None), ('local_pref', None), ('route_targets', None), ('import_targets', None), ('export_targets', None), ('route_distinguishers', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        cols, data = self.cmd.take_action(parsed_args)
        self.networkclient.create_bgpvpn.assert_called_once_with(**{'type': 'l3'})
        self.assertEqual(sorted(sorted_columns), sorted(cols))

    def test_create_bgpvpn_with_all_args(self):
        attrs = {'tenant_id': 'new_fake_project_id', 'name': 'fake_name', 'type': 'l2', 'vni': 100, 'local_pref': 777, 'route_targets': ['fake_rt1', 'fake_rt2', 'fake_rt3'], 'import_targets': ['fake_irt1', 'fake_irt2', 'fake_irt3'], 'export_targets': ['fake_ert1', 'fake_ert2', 'fake_ert3'], 'route_distinguishers': ['fake_rd1', 'fake_rd2', 'fake_rd3']}
        fake_bgpvpn = fakes.create_one_bgpvpn(attrs)
        self.networkclient.create_bgpvpn = mock.Mock(return_value=fake_bgpvpn)
        arglist = ['--project', fake_bgpvpn['tenant_id'], '--name', fake_bgpvpn['name'], '--type', fake_bgpvpn['type'], '--vni', str(fake_bgpvpn['vni']), '--local-pref', str(fake_bgpvpn['local_pref'])]
        for rt in fake_bgpvpn['route_targets']:
            arglist.extend(['--route-target', rt])
        for rt in fake_bgpvpn['import_targets']:
            arglist.extend(['--import-target', rt])
        for rt in fake_bgpvpn['export_targets']:
            arglist.extend(['--export-target', rt])
        for rd in fake_bgpvpn['route_distinguishers']:
            arglist.extend(['--route-distinguisher', rd])
        verifylist = [('project', fake_bgpvpn['tenant_id']), ('name', fake_bgpvpn['name']), ('type', fake_bgpvpn['type']), ('vni', fake_bgpvpn['vni']), ('local_pref', fake_bgpvpn['local_pref']), ('route_targets', fake_bgpvpn['route_targets']), ('import_targets', fake_bgpvpn['import_targets']), ('export_targets', fake_bgpvpn['export_targets']), ('route_distinguishers', fake_bgpvpn['route_distinguishers'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        cols, data = self.cmd.take_action(parsed_args)
        fake_bgpvpn_call = copy.deepcopy(attrs)
        self.networkclient.create_bgpvpn.assert_called_once_with(**fake_bgpvpn_call)
        self.assertEqual(sorted(sorted_columns), sorted(cols))