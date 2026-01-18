from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_networks as osc_share_networks
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.ddt
class TestShareNetworkSet(TestShareNetwork):

    def setUp(self):
        super(TestShareNetworkSet, self).setUp()
        self.share_network = manila_fakes.FakeShareNetwork.create_one_share_network()
        self.share_networks_mock.get.return_value = self.share_network
        self.cmd = osc_share_networks.SetShareNetwork(self.app, None)

    @ddt.data({'status': 'error', 'current_security_service': str(uuid.uuid4()), 'check_only': True, 'restart_check': True}, {'status': None, 'current_security_service': str(uuid.uuid4()), 'check_only': True, 'restart_check': None}, {'status': None, 'current_security_service': str(uuid.uuid4()), 'check_only': True, 'restart_check': True})
    @ddt.unpack
    def test_set_share_network_api_version_exception(self, status, current_security_service, check_only, restart_check):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.62')
        arglist = [self.share_network.id]
        verifylist = [('share_network', self.share_network.id)]
        if status:
            arglist.extend(['--status', status])
            verifylist.append(('status', status))
        if current_security_service:
            arglist.extend(['--current-security-service', current_security_service])
            verifylist.append(('current_security_service', current_security_service))
        if check_only and restart_check:
            arglist.extend(['--check-only', '--restart-check'])
            verifylist.extend([('check_only', True), ('restart_check', True)])
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_set_network_properties(self):
        new_name = 'share-network-name-' + uuid.uuid4().hex
        new_description = 'share-network-description-' + uuid.uuid4().hex
        new_neutron_subnet_id = str(uuid.uuid4())
        arglist = [self.share_network.id, '--name', new_name, '--description', new_description, '--neutron-subnet-id', new_neutron_subnet_id]
        verifylist = [('share_network', self.share_network.id), ('name', new_name), ('description', new_description), ('neutron_subnet_id', new_neutron_subnet_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.find_resource', return_value=self.share_network):
            result = self.cmd.take_action(parsed_args)
        self.share_networks_mock.update.assert_called_once_with(self.share_network, name=parsed_args.name, description=new_description, neutron_subnet_id=new_neutron_subnet_id)
        self.assertIsNone(result)

    def test_set_share_network_status(self):
        arglist = [self.share_network.id, '--status', 'error']
        verifylist = [('share_network', self.share_network.id), ('status', 'error')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.find_resource', return_value=self.share_network):
            result = self.cmd.take_action(parsed_args)
        self.share_networks_mock.reset_state.assert_called_once_with(self.share_network, parsed_args.status)
        self.assertIsNone(result)

    def test_set_network_update_exception(self):
        share_network_name = 'share-network-name-' + uuid.uuid4().hex
        arglist = [self.share_network.id, '--name', share_network_name]
        verifylist = [('share_network', self.share_network.id), ('name', share_network_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.share_networks_mock.update.side_effect = Exception()
        with mock.patch('osc_lib.utils.find_resource', return_value=self.share_network):
            self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.share_networks_mock.update.assert_called_once_with(self.share_network, name=parsed_args.name)

    def test_set_share_network_status_exception(self):
        arglist = [self.share_network.id, '--status', 'error']
        verifylist = [('share_network', self.share_network.id), ('status', 'error')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.share_networks_mock.reset_state.side_effect = Exception()
        with mock.patch('osc_lib.utils.find_resource', return_value=self.share_network):
            self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.share_networks_mock.reset_state.assert_called_once_with(self.share_network, parsed_args.status)

    @ddt.data({'check_only': False, 'restart_check': False}, {'check_only': True, 'restart_check': True}, {'check_only': True, 'restart_check': False})
    @ddt.unpack
    def test_set_share_network_add_new_security_service_check_reset(self, check_only, restart_check):
        self.share_networks_mock.add_security_service_check = mock.Mock(return_value=(200, {'compatible': True}))
        arglist = [self.share_network.id, '--new-security-service', 'new-security-service-name']
        verifylist = [('share_network', self.share_network.id), ('new_security_service', 'new-security-service-name')]
        if check_only:
            arglist.append('--check-only')
            verifylist.append(('check_only', True))
        if restart_check:
            arglist.append('--restart-check')
            verifylist.append(('restart_check', True))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.find_resource', side_effect=[self.share_network, 'new-security-service']):
            result = self.cmd.take_action(parsed_args)
        if check_only:
            self.share_networks_mock.add_security_service_check.assert_called_once_with(self.share_network, 'new-security-service', reset_operation=restart_check)
            self.share_networks_mock.add_security_service.assert_not_called()
        else:
            self.share_networks_mock.add_security_service_check.assert_not_called()
            self.share_networks_mock.add_security_service.assert_called_once_with(self.share_network, 'new-security-service')
        self.assertIsNone(result)

    @ddt.data({'check_only': False, 'restart_check': False}, {'check_only': True, 'restart_check': True}, {'check_only': True, 'restart_check': False})
    @ddt.unpack
    def test_set_share_network_update_security_service_check_reset(self, check_only, restart_check):
        self.share_networks_mock.update_share_network_security_service_check = mock.Mock(return_value=(200, {'compatible': True}))
        arglist = [self.share_network.id, '--new-security-service', 'new-security-service-name', '--current-security-service', 'current-security-service-name']
        verifylist = [('share_network', self.share_network.id), ('new_security_service', 'new-security-service-name'), ('current_security_service', 'current-security-service-name')]
        if check_only:
            arglist.append('--check-only')
            verifylist.append(('check_only', True))
        if restart_check:
            arglist.append('--restart-check')
            verifylist.append(('restart_check', True))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.find_resource', side_effect=[self.share_network, 'new-security-service', 'current-security-service']):
            result = self.cmd.take_action(parsed_args)
        if check_only:
            self.share_networks_mock.update_share_network_security_service_check.assert_called_once_with(self.share_network, 'current-security-service', 'new-security-service', reset_operation=restart_check)
            self.share_networks_mock.update_share_network_security_service.assert_not_called()
        else:
            self.share_networks_mock.update_share_network_security_service_check.assert_not_called()
            self.share_networks_mock.update_share_network_security_service.assert_called_once_with(self.share_network, 'current-security-service', 'new-security-service')
        self.assertIsNone(result)