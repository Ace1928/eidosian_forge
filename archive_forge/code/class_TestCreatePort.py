from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.network.v2 import port
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
class TestCreatePort(TestPort):
    _port = network_fakes.create_one_port()
    columns, data = TestPort._get_common_cols_data(_port)

    def setUp(self):
        super(TestCreatePort, self).setUp()
        self.network_client.create_port = mock.Mock(return_value=self._port)
        self.network_client.set_tags = mock.Mock(return_value=None)
        fake_net = network_fakes.create_one_network({'id': self._port.network_id})
        self.network_client.find_network = mock.Mock(return_value=fake_net)
        self.fake_subnet = network_fakes.FakeSubnet.create_one_subnet()
        self.network_client.find_subnet = mock.Mock(return_value=self.fake_subnet)
        self.network_client.find_extension = mock.Mock(return_value=[])
        self.cmd = port.CreatePort(self.app, self.namespace)

    def test_create_default_options(self):
        arglist = ['--network', self._port.network_id, 'test-port']
        verifylist = [('network', self._port.network_id), ('enable', True), ('name', 'test-port')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_port.assert_called_once_with(**{'admin_state_up': True, 'network_id': self._port.network_id, 'name': 'test-port'})
        self.assertFalse(self.network_client.set_tags.called)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_full_options(self):
        arglist = ['--mac-address', 'aa:aa:aa:aa:aa:aa', '--fixed-ip', 'subnet=%s,ip-address=10.0.0.2' % self.fake_subnet.id, '--description', self._port.description, '--device', 'deviceid', '--device-owner', 'fakeowner', '--disable', '--vnic-type', 'macvtap', '--binding-profile', 'foo=bar', '--binding-profile', 'foo2=bar2', '--network', self._port.network_id, '--dns-domain', 'example.org', '--dns-name', '8.8.8.8', 'test-port']
        verifylist = [('mac_address', 'aa:aa:aa:aa:aa:aa'), ('fixed_ip', [{'subnet': self.fake_subnet.id, 'ip-address': '10.0.0.2'}]), ('description', self._port.description), ('device', 'deviceid'), ('device_owner', 'fakeowner'), ('disable', True), ('vnic_type', 'macvtap'), ('binding_profile', {'foo': 'bar', 'foo2': 'bar2'}), ('network', self._port.network_id), ('dns_domain', 'example.org'), ('dns_name', '8.8.8.8'), ('name', 'test-port')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_port.assert_called_once_with(**{'mac_address': 'aa:aa:aa:aa:aa:aa', 'fixed_ips': [{'subnet_id': self.fake_subnet.id, 'ip_address': '10.0.0.2'}], 'description': self._port.description, 'device_id': 'deviceid', 'device_owner': 'fakeowner', 'admin_state_up': False, 'binding:vnic_type': 'macvtap', 'binding:profile': {'foo': 'bar', 'foo2': 'bar2'}, 'network_id': self._port.network_id, 'dns_domain': 'example.org', 'dns_name': '8.8.8.8', 'name': 'test-port'})
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_invalid_json_binding_profile(self):
        arglist = ['--network', self._port.network_id, '--binding-profile', '{"parent_name":"fake_parent"', 'test-port']
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, None)

    def test_create_invalid_key_value_binding_profile(self):
        arglist = ['--network', self._port.network_id, '--binding-profile', 'key', 'test-port']
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, None)

    def test_create_json_binding_profile(self):
        arglist = ['--network', self._port.network_id, '--binding-profile', '{"parent_name":"fake_parent"}', '--binding-profile', '{"tag":42}', 'test-port']
        verifylist = [('network', self._port.network_id), ('enable', True), ('binding_profile', {'parent_name': 'fake_parent', 'tag': 42}), ('name', 'test-port')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_port.assert_called_once_with(**{'admin_state_up': True, 'network_id': self._port.network_id, 'binding:profile': {'parent_name': 'fake_parent', 'tag': 42}, 'name': 'test-port'})
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_with_security_group(self):
        secgroup = network_fakes.FakeSecurityGroup.create_one_security_group()
        self.network_client.find_security_group = mock.Mock(return_value=secgroup)
        arglist = ['--network', self._port.network_id, '--security-group', secgroup.id, 'test-port']
        verifylist = [('network', self._port.network_id), ('enable', True), ('security_group', [secgroup.id]), ('name', 'test-port')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_port.assert_called_once_with(**{'admin_state_up': True, 'network_id': self._port.network_id, 'security_group_ids': [secgroup.id], 'name': 'test-port'})
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_port_with_dns_name(self):
        arglist = ['--network', self._port.network_id, '--dns-name', '8.8.8.8', 'test-port']
        verifylist = [('network', self._port.network_id), ('enable', True), ('dns_name', '8.8.8.8'), ('name', 'test-port')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_port.assert_called_once_with(**{'admin_state_up': True, 'network_id': self._port.network_id, 'dns_name': '8.8.8.8', 'name': 'test-port'})
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_with_security_groups(self):
        sg_1 = network_fakes.FakeSecurityGroup.create_one_security_group()
        sg_2 = network_fakes.FakeSecurityGroup.create_one_security_group()
        self.network_client.find_security_group = mock.Mock(side_effect=[sg_1, sg_2])
        arglist = ['--network', self._port.network_id, '--security-group', sg_1.id, '--security-group', sg_2.id, 'test-port']
        verifylist = [('network', self._port.network_id), ('enable', True), ('security_group', [sg_1.id, sg_2.id]), ('name', 'test-port')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_port.assert_called_once_with(**{'admin_state_up': True, 'network_id': self._port.network_id, 'security_group_ids': [sg_1.id, sg_2.id], 'name': 'test-port'})
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_with_no_security_groups(self):
        arglist = ['--network', self._port.network_id, '--no-security-group', 'test-port']
        verifylist = [('network', self._port.network_id), ('enable', True), ('no_security_group', True), ('name', 'test-port')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_port.assert_called_once_with(**{'admin_state_up': True, 'network_id': self._port.network_id, 'security_group_ids': [], 'name': 'test-port'})
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_with_no_fixed_ips(self):
        arglist = ['--network', self._port.network_id, '--no-fixed-ip', 'test-port']
        verifylist = [('network', self._port.network_id), ('enable', True), ('no_fixed_ip', True), ('name', 'test-port')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_port.assert_called_once_with(**{'admin_state_up': True, 'network_id': self._port.network_id, 'fixed_ips': [], 'name': 'test-port'})
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_port_with_allowed_address_pair_ipaddr(self):
        pairs = [{'ip_address': '192.168.1.123'}, {'ip_address': '192.168.1.45'}]
        arglist = ['--network', self._port.network_id, '--allowed-address', 'ip-address=192.168.1.123', '--allowed-address', 'ip-address=192.168.1.45', 'test-port']
        verifylist = [('network', self._port.network_id), ('enable', True), ('allowed_address_pairs', [{'ip-address': '192.168.1.123'}, {'ip-address': '192.168.1.45'}]), ('name', 'test-port')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_port.assert_called_once_with(**{'admin_state_up': True, 'network_id': self._port.network_id, 'allowed_address_pairs': pairs, 'name': 'test-port'})
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_port_with_allowed_address_pair(self):
        pairs = [{'ip_address': '192.168.1.123', 'mac_address': 'aa:aa:aa:aa:aa:aa'}, {'ip_address': '192.168.1.45', 'mac_address': 'aa:aa:aa:aa:aa:b1'}]
        arglist = ['--network', self._port.network_id, '--allowed-address', 'ip-address=192.168.1.123,mac-address=aa:aa:aa:aa:aa:aa', '--allowed-address', 'ip-address=192.168.1.45,mac-address=aa:aa:aa:aa:aa:b1', 'test-port']
        verifylist = [('network', self._port.network_id), ('enable', True), ('allowed_address_pairs', [{'ip-address': '192.168.1.123', 'mac-address': 'aa:aa:aa:aa:aa:aa'}, {'ip-address': '192.168.1.45', 'mac-address': 'aa:aa:aa:aa:aa:b1'}]), ('name', 'test-port')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_port.assert_called_once_with(**{'admin_state_up': True, 'network_id': self._port.network_id, 'allowed_address_pairs': pairs, 'name': 'test-port'})
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_port_with_qos(self):
        qos_policy = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy()
        self.network_client.find_qos_policy = mock.Mock(return_value=qos_policy)
        arglist = ['--network', self._port.network_id, '--qos-policy', qos_policy.id, 'test-port']
        verifylist = [('network', self._port.network_id), ('enable', True), ('qos_policy', qos_policy.id), ('name', 'test-port')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_port.assert_called_once_with(**{'admin_state_up': True, 'network_id': self._port.network_id, 'qos_policy_id': qos_policy.id, 'name': 'test-port'})
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_port_security_enabled(self):
        arglist = ['--network', self._port.network_id, '--enable-port-security', 'test-port']
        verifylist = [('network', self._port.network_id), ('enable', True), ('enable_port_security', True), ('name', 'test-port')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.network_client.create_port.assert_called_once_with(**{'admin_state_up': True, 'network_id': self._port.network_id, 'port_security_enabled': True, 'name': 'test-port'})

    def test_create_port_security_disabled(self):
        arglist = ['--network', self._port.network_id, '--disable-port-security', 'test-port']
        verifylist = [('network', self._port.network_id), ('enable', True), ('disable_port_security', True), ('name', 'test-port')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.network_client.create_port.assert_called_once_with(**{'admin_state_up': True, 'network_id': self._port.network_id, 'port_security_enabled': False, 'name': 'test-port'})

    def _test_create_with_tag(self, add_tags=True, add_tags_in_post=True):
        arglist = ['--network', self._port.network_id, 'test-port']
        if add_tags:
            arglist += ['--tag', 'red', '--tag', 'blue']
        else:
            arglist += ['--no-tag']
        verifylist = [('network', self._port.network_id), ('enable', True), ('name', 'test-port')]
        if add_tags:
            verifylist.append(('tags', ['red', 'blue']))
        else:
            verifylist.append(('no_tag', True))
        self.network_client.find_extension = mock.Mock(return_value=add_tags_in_post)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        args = {'admin_state_up': True, 'network_id': self._port.network_id, 'name': 'test-port'}
        if add_tags_in_post:
            if add_tags:
                args['tags'] = sorted(['red', 'blue'])
            else:
                args['tags'] = []
            self.network_client.create_port.assert_called_once()
            create_port_call_kwargs = self.network_client.create_port.call_args[1]
            create_port_call_kwargs['tags'] = sorted(create_port_call_kwargs['tags'])
            self.assertDictEqual(args, create_port_call_kwargs)
        else:
            self.network_client.create_port.assert_called_once_with(admin_state_up=True, network_id=self._port.network_id, name='test-port')
            if add_tags:
                self.network_client.set_tags.assert_called_once_with(self._port, test_utils.CompareBySet(['red', 'blue']))
            else:
                self.assertFalse(self.network_client.set_tags.called)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_with_tags(self):
        self._test_create_with_tag(add_tags=True, add_tags_in_post=True)

    def test_create_with_no_tag(self):
        self._test_create_with_tag(add_tags=False, add_tags_in_post=True)

    def test_create_with_tags_using_put(self):
        self._test_create_with_tag(add_tags=True, add_tags_in_post=False)

    def test_create_with_no_tag_using_put(self):
        self._test_create_with_tag(add_tags=False, add_tags_in_post=False)

    def _test_create_with_uplink_status_propagation(self, enable=True):
        arglist = ['--network', self._port.network_id, 'test-port']
        if enable:
            arglist += ['--enable-uplink-status-propagation']
        else:
            arglist += ['--disable-uplink-status-propagation']
        verifylist = [('network', self._port.network_id), ('name', 'test-port')]
        if enable:
            verifylist.append(('enable_uplink_status_propagation', True))
        else:
            verifylist.append(('disable_uplink_status_propagation', True))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_port.assert_called_once_with(**{'admin_state_up': True, 'network_id': self._port.network_id, 'propagate_uplink_status': enable, 'name': 'test-port'})
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_with_uplink_status_propagation_enabled(self):
        self._test_create_with_uplink_status_propagation(enable=True)

    def test_create_with_uplink_status_propagation_disabled(self):
        self._test_create_with_uplink_status_propagation(enable=False)

    def test_create_port_with_extra_dhcp_option(self):
        extra_dhcp_options = [{'opt_name': 'classless-static-route', 'opt_value': '169.254.169.254/32,22.2.0.2,0.0.0.0/0,22.2.0.1', 'ip_version': '4'}, {'opt_name': 'dns-server', 'opt_value': '240C::6666', 'ip_version': '6'}]
        arglist = ['--network', self._port.network_id, '--extra-dhcp-option', 'name=classless-static-route,value=169.254.169.254/32,22.2.0.2,0.0.0.0/0,22.2.0.1,ip-version=4', '--extra-dhcp-option', 'name=dns-server,value=240C::6666,ip-version=6', 'test-port']
        verifylist = [('network', self._port.network_id), ('extra_dhcp_options', [{'name': 'classless-static-route', 'value': '169.254.169.254/32,22.2.0.2,0.0.0.0/0,22.2.0.1', 'ip-version': '4'}, {'name': 'dns-server', 'value': '240C::6666', 'ip-version': '6'}]), ('name', 'test-port')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.network_client.create_port.assert_called_once_with(**{'admin_state_up': True, 'network_id': self._port.network_id, 'extra_dhcp_opts': extra_dhcp_options, 'name': 'test-port'})

    def _test_create_with_numa_affinity_policy(self, policy=None):
        arglist = ['--network', self._port.network_id, 'test-port']
        if policy:
            arglist += ['--numa-policy-%s' % policy]
        numa_affinity_policy = None if not policy else policy
        verifylist = [('network', self._port.network_id), ('name', 'test-port')]
        if policy:
            verifylist.append(('numa_policy_%s' % policy, True))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        create_args = {'admin_state_up': True, 'network_id': self._port.network_id, 'name': 'test-port'}
        if numa_affinity_policy:
            create_args['numa_affinity_policy'] = numa_affinity_policy
        self.network_client.create_port.assert_called_once_with(**create_args)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_with_numa_affinity_policy_required(self):
        self._test_create_with_numa_affinity_policy(policy='required')

    def test_create_with_numa_affinity_policy_preferred(self):
        self._test_create_with_numa_affinity_policy(policy='preferred')

    def test_create_with_numa_affinity_policy_legacy(self):
        self._test_create_with_numa_affinity_policy(policy='legacy')

    def test_create_with_numa_affinity_policy_null(self):
        self._test_create_with_numa_affinity_policy()

    def test_create_with_device_profile(self):
        arglist = ['--network', self._port.network_id, '--device-profile', 'cyborg_device_profile_1', 'test-port']
        verifylist = [('network', self._port.network_id), ('device_profile', self._port.device_profile), ('name', 'test-port')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        create_args = {'admin_state_up': True, 'network_id': self._port.network_id, 'name': 'test-port', 'device_profile': 'cyborg_device_profile_1'}
        self.network_client.create_port.assert_called_once_with(**create_args)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_hints_invalid_json(self):
        arglist = ['--network', self._port.network_id, '--hint', 'invalid json', 'test-port']
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, None)

    def test_create_hints_invalid_alias(self):
        arglist = ['--network', self._port.network_id, '--hint', 'invalid-alias=value', 'test-port']
        verifylist = [('network', self._port.network_id), ('enable', True), ('hint', {'invalid-alias': 'value'}), ('name', 'test-port')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_create_hints_invalid_value(self):
        arglist = ['--network', self._port.network_id, '--hint', 'ovs-tx-steering=invalid-value', 'test-port']
        verifylist = [('network', self._port.network_id), ('enable', True), ('hint', {'ovs-tx-steering': 'invalid-value'}), ('name', 'test-port')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_create_hints_valid_alias_value(self):
        arglist = ['--network', self._port.network_id, '--hint', 'ovs-tx-steering=hash', 'test-port']
        verifylist = [('network', self._port.network_id), ('enable', True), ('hint', {'ovs-tx-steering': 'hash'}), ('name', 'test-port')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_port.assert_called_once_with(**{'admin_state_up': True, 'network_id': self._port.network_id, 'hints': {'openvswitch': {'other_config': {'tx-steering': 'hash'}}}, 'name': 'test-port'})
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_hints_valid_json(self):
        arglist = ['--network', self._port.network_id, '--hint', '{"openvswitch": {"other_config": {"tx-steering": "hash"}}}', 'test-port']
        verifylist = [('network', self._port.network_id), ('enable', True), ('hint', {'openvswitch': {'other_config': {'tx-steering': 'hash'}}}), ('name', 'test-port')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_port.assert_called_once_with(**{'admin_state_up': True, 'network_id': self._port.network_id, 'hints': {'openvswitch': {'other_config': {'tx-steering': 'hash'}}}, 'name': 'test-port'})
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def _test_create_with_hardware_offload_type(self, hwol_type=None):
        arglist = ['--network', self._port.network_id, 'test-port']
        if hwol_type:
            arglist += ['--hardware-offload-type', hwol_type]
        hardware_offload_type = None if not hwol_type else hwol_type
        verifylist = [('network', self._port.network_id), ('name', 'test-port')]
        if hwol_type:
            verifylist.append(('hardware_offload_type', hwol_type))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        create_args = {'admin_state_up': True, 'network_id': self._port.network_id, 'name': 'test-port'}
        if hwol_type:
            create_args['hardware_offload_type'] = hardware_offload_type
        self.network_client.create_port.assert_called_once_with(**create_args)
        self.assertEqual(set(self.columns), set(columns))
        self.assertCountEqual(self.data, data)

    def test_create_with_hardware_offload_type_switchdev(self):
        self._test_create_with_hardware_offload_type(hwol_type='switchdev')

    def test_create_with_hardware_offload_type_null(self):
        self._test_create_with_hardware_offload_type()