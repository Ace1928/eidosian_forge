from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from manilaclient import api_versions
from manilaclient.osc.v2 import share_network_subnets as osc_share_subnets
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.ddt
class TestShareNetworkSubnetCreate(TestShareNetworkSubnet):

    def setUp(self):
        super(TestShareNetworkSubnetCreate, self).setUp()
        self.share_network = manila_fakes.FakeShareNetwork.create_one_share_network()
        self.share_networks_mock.get.return_value = self.share_network
        self.share_network_subnet = manila_fakes.FakeShareNetworkSubnet.create_one_share_subnet()
        self.share_subnets_mock.create.return_value = self.share_network_subnet
        self.cmd = osc_share_subnets.CreateShareNetworkSubnet(self.app, None)
        self.data = self.share_network_subnet._info.values()
        self.columns = self.share_network_subnet._info.keys()

    def test_share_network_subnet_create_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_network_subnet_create(self):
        fake_neutron_net_id = str(uuid.uuid4())
        fake_neutron_subnet_id = str(uuid.uuid4())
        arglist = [self.share_network.id, '--neutron-net-id', fake_neutron_net_id, '--neutron-subnet-id', fake_neutron_subnet_id, '--availability-zone', 'nova']
        verifylist = [('share_network', self.share_network.id), ('neutron_net_id', fake_neutron_net_id), ('neutron_subnet_id', fake_neutron_subnet_id), ('availability_zone', 'nova')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.share_subnets_mock.create.assert_called_once_with(neutron_net_id=fake_neutron_net_id, neutron_subnet_id=fake_neutron_subnet_id, availability_zone='nova', share_network_id=self.share_network.id, metadata={})
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_share_network_subnet_create_arg_group_exception(self):
        fake_neutron_net_id = str(uuid.uuid4())
        arglist = [self.share_network.id, '--neutron-net-id', fake_neutron_net_id]
        verifylist = [('share_network', self.share_network.id), ('neutron_net_id', fake_neutron_net_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    @ddt.data({'check_only': False, 'restart_check': True}, {'check_only': True, 'restart_check': True}, {'check_only': True, 'restart_check': False})
    @ddt.unpack
    def test_share_network_subnet_create_check_api_version_exception(self, check_only, restart_check):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.69')
        arglist = [self.share_network.id]
        verifylist = [('share_network', self.share_network.id)]
        if check_only:
            arglist.append('--check-only')
            verifylist.append(('check_only', True))
        if restart_check:
            arglist.append('--restart-check')
            verifylist.append(('restart_check', True))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    @ddt.data(True, False)
    def test_share_network_subnet_create_check(self, restart_check):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.70')
        self.share_networks_mock.share_network_subnet_create_check = mock.Mock(return_value=(200, {'compatible': True}))
        arglist = [self.share_network.id, '--check-only']
        verifylist = [('share_network', self.share_network.id), ('check_only', True)]
        if restart_check:
            arglist.append('--restart-check')
            verifylist.append(('restart_check', True))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.share_networks_mock.share_network_subnet_create_check.assert_called_once_with(share_network_id=self.share_network.id, neutron_net_id=None, neutron_subnet_id=None, availability_zone=None, reset_operation=restart_check)

    def test_share_network_subnet_create_metadata(self):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.78')
        arglist = [self.share_network.id, '--property', 'Manila=zorilla', '--property', 'Zorilla=manila']
        verifylist = [('share_network', self.share_network.id), ('property', {'Manila': 'zorilla', 'Zorilla': 'manila'})]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.share_subnets_mock.create.assert_called_once_with(neutron_net_id=None, neutron_subnet_id=None, availability_zone=None, share_network_id=self.share_network.id, metadata={'Manila': 'zorilla', 'Zorilla': 'manila'})
        self.assertEqual(set(self.columns), set(columns))
        self.assertCountEqual(self.data, data)

    def test_share_network_subnet_create_metadata_api_version_exception(self):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.77')
        arglist = [self.share_network.id, '--property', 'Manila=zorilla']
        verifylist = [('share_network', self.share_network.id), ('property', {'Manila': 'zorilla'})]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)