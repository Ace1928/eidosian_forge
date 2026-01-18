import collections
import copy
import getpass
import json
import tempfile
from unittest import mock
from unittest.mock import call
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
class TestServerCreate(TestServer):
    columns = ('OS-EXT-STS:power_state', 'addresses', 'flavor', 'id', 'image', 'name', 'properties')

    def datalist(self):
        datalist = (server.PowerStateColumn(getattr(self.new_server, 'OS-EXT-STS:power_state')), format_columns.DictListColumn({}), self.flavor.name + ' (' + self.new_server.flavor.get('id') + ')', self.new_server.id, self.image.name + ' (' + self.new_server.image.get('id') + ')', self.new_server.name, format_columns.DictColumn(self.new_server.metadata))
        return datalist

    def setUp(self):
        super(TestServerCreate, self).setUp()
        attrs = {'networks': {}}
        self.new_server = compute_fakes.create_one_server(attrs=attrs)
        self.servers_mock.get.return_value = self.new_server
        self.servers_mock.create.return_value = self.new_server
        self.image = image_fakes.create_one_image()
        self.image_client.find_image.return_value = self.image
        self.image_client.get_image.return_value = self.image
        self.flavor = compute_fakes.create_one_flavor()
        self.flavors_mock.get.return_value = self.flavor
        self.volume = volume_fakes.create_one_volume()
        self.volume_alt = volume_fakes.create_one_volume()
        self.volumes_mock.get.return_value = self.volume
        self.snapshot = volume_fakes.create_one_snapshot()
        self.snapshots_mock.get.return_value = self.snapshot
        self.cmd = server.CreateServer(self.app, None)

    def test_server_create_no_options(self):
        arglist = [self.new_server.name]
        verifylist = [('server_name', self.new_server.name)]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_server_create_minimal(self):
        arglist = ['--image', 'image1', '--flavor', 'flavor1', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[], nics=[], scheduler_hints={}, config_drive=None)
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)
        self.assertFalse(self.image_client.images.called)
        self.assertFalse(self.flavors_mock.called)

    def test_server_create_with_options(self):
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--key-name', 'keyname', '--property', 'Beta=b', '--security-group', 'securitygroup', '--use-config-drive', '--password', 'passw0rd', '--hint', 'a=b', '--hint', 'a=c', '--server-group', 'servergroup', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('key_name', 'keyname'), ('properties', {'Beta': 'b'}), ('security_group', ['securitygroup']), ('hints', {'a': ['b', 'c']}), ('server_group', 'servergroup'), ('config_drive', True), ('password', 'passw0rd'), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        fake_server_group = compute_fakes.create_one_server_group()
        self.compute_client.server_groups.get.return_value = fake_server_group
        fake_sg = network_fakes.FakeSecurityGroup.create_security_groups()
        mock_find_sg = network_fakes.FakeSecurityGroup.get_security_groups(fake_sg)
        self.app.client_manager.network.find_security_group = mock_find_sg
        columns, data = self.cmd.take_action(parsed_args)
        mock_find_sg.assert_called_once_with('securitygroup', ignore_missing=False)
        kwargs = dict(meta={'Beta': 'b'}, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[fake_sg[0].id], userdata=None, key_name='keyname', availability_zone=None, admin_pass='passw0rd', block_device_mapping_v2=[], nics=[], scheduler_hints={'a': ['b', 'c'], 'group': fake_server_group.id}, config_drive=True)
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)

    def test_server_create_with_not_exist_security_group(self):
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--key-name', 'keyname', '--security-group', 'securitygroup', '--security-group', 'not_exist_sg', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('key_name', 'keyname'), ('security_group', ['securitygroup', 'not_exist_sg']), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        fake_sg = network_fakes.FakeSecurityGroup.create_security_groups(count=1)
        fake_sg.append(exceptions.NotFound(code=404))
        mock_find_sg = network_fakes.FakeSecurityGroup.get_security_groups(fake_sg)
        self.app.client_manager.network.find_security_group = mock_find_sg
        self.assertRaises(exceptions.NotFound, self.cmd.take_action, parsed_args)
        mock_find_sg.assert_called_with('not_exist_sg', ignore_missing=False)

    def test_server_create_with_security_group_in_nova_network(self):
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--key-name', 'keyname', '--security-group', 'securitygroup', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('key_name', 'keyname'), ('security_group', ['securitygroup']), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(self.app.client_manager, 'is_network_endpoint_enabled', return_value=False):
            with mock.patch.object(self.compute_client.api, 'security_group_find', return_value={'name': 'fake_sg'}) as mock_find:
                columns, data = self.cmd.take_action(parsed_args)
                mock_find.assert_called_once_with('securitygroup')
        kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=['fake_sg'], userdata=None, key_name='keyname', availability_zone=None, admin_pass=None, block_device_mapping_v2=[], nics=[], scheduler_hints={}, config_drive=None)
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)

    def test_server_create_with_network(self):
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--network', 'net1', '--nic', 'net-id=net1,v4-fixed-ip=10.0.0.2', '--port', 'port1', '--network', 'net1', '--network', 'auto', '--nic', 'port-id=port2', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('nics', [{'net-id': 'net1', 'port-id': '', 'v4-fixed-ip': '', 'v6-fixed-ip': ''}, {'net-id': 'net1', 'port-id': '', 'v4-fixed-ip': '10.0.0.2', 'v6-fixed-ip': ''}, {'net-id': '', 'port-id': 'port1', 'v4-fixed-ip': '', 'v6-fixed-ip': ''}, {'net-id': 'net1', 'port-id': '', 'v4-fixed-ip': '', 'v6-fixed-ip': ''}, {'net-id': 'auto', 'port-id': '', 'v4-fixed-ip': '', 'v6-fixed-ip': ''}, {'net-id': '', 'port-id': 'port2', 'v4-fixed-ip': '', 'v6-fixed-ip': ''}]), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        get_endpoints = mock.Mock()
        get_endpoints.return_value = {'network': []}
        self.app.client_manager.auth_ref = mock.Mock()
        self.app.client_manager.auth_ref.service_catalog = mock.Mock()
        self.app.client_manager.auth_ref.service_catalog.get_endpoints = get_endpoints
        network_resource = mock.Mock(id='net1_uuid')
        port1_resource = mock.Mock(id='port1_uuid')
        port2_resource = mock.Mock(id='port2_uuid')
        self.network_client.find_network.return_value = network_resource
        self.network_client.find_port.side_effect = lambda port_id, ignore_missing: {'port1': port1_resource, 'port2': port2_resource}[port_id]
        _network_1 = mock.Mock(id='net1_uuid')
        _network_auto = mock.Mock(id='auto_uuid')
        _port1 = mock.Mock(id='port1_uuid')
        _port2 = mock.Mock(id='port2_uuid')
        find_network = mock.Mock()
        find_port = mock.Mock()
        find_network.side_effect = lambda net_id, ignore_missing: {'net1': _network_1, 'auto': _network_auto}[net_id]
        find_port.side_effect = lambda port_id, ignore_missing: {'port1': _port1, 'port2': _port2}[port_id]
        self.app.client_manager.network.find_network = find_network
        self.app.client_manager.network.find_port = find_port
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[], nics=[{'net-id': 'net1_uuid', 'v4-fixed-ip': '', 'v6-fixed-ip': '', 'port-id': ''}, {'net-id': 'net1_uuid', 'v4-fixed-ip': '10.0.0.2', 'v6-fixed-ip': '', 'port-id': ''}, {'net-id': '', 'v4-fixed-ip': '', 'v6-fixed-ip': '', 'port-id': 'port1_uuid'}, {'net-id': 'net1_uuid', 'v4-fixed-ip': '', 'v6-fixed-ip': '', 'port-id': ''}, {'net-id': 'auto_uuid', 'v4-fixed-ip': '', 'v6-fixed-ip': '', 'port-id': ''}, {'net-id': '', 'v4-fixed-ip': '', 'v6-fixed-ip': '', 'port-id': 'port2_uuid'}], scheduler_hints={}, config_drive=None)
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)

    def test_server_create_with_network_tag(self):
        self.compute_client.api_version = api_versions.APIVersion('2.43')
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--nic', 'net-id=net1,tag=foo', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('nics', [{'net-id': 'net1', 'port-id': '', 'v4-fixed-ip': '', 'v6-fixed-ip': '', 'tag': 'foo'}]), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        _network = mock.Mock(id='net1_uuid')
        self.network_client.find_network.return_value = _network
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[], nics=[{'net-id': 'net1_uuid', 'v4-fixed-ip': '', 'v6-fixed-ip': '', 'port-id': '', 'tag': 'foo'}], scheduler_hints={}, config_drive=None)
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)
        self.network_client.find_network.assert_called_once()
        self.app.client_manager.network.find_network.assert_called_once()

    def test_server_create_with_network_tag_pre_v243(self):
        self.compute_client.api_version = api_versions.APIVersion('2.42')
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--nic', 'net-id=net1,tag=foo', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('nics', [{'net-id': 'net1', 'port-id': '', 'v4-fixed-ip': '', 'v6-fixed-ip': '', 'tag': 'foo'}]), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def _test_server_create_with_auto_network(self, arglist):
        self.compute_client.api_version = api_versions.APIVersion('2.37')
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('nics', ['auto']), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[], nics='auto', scheduler_hints={}, config_drive=None)
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)

    def test_server_create_with_auto_network_legacy(self):
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--nic', 'auto', self.new_server.name]
        self._test_server_create_with_auto_network(arglist)

    def test_server_create_with_auto_network(self):
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--auto-network', self.new_server.name]
        self._test_server_create_with_auto_network(arglist)

    def test_server_create_with_auto_network_pre_v237(self):
        self.compute_client.api_version = api_versions.APIVersion('2.36')
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--nic', 'auto', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('nics', ['auto']), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.37 or greater is required to support explicit auto-allocation of a network or to disable network allocation', str(exc))
        self.assertNotCalled(self.servers_mock.create)

    def test_server_create_with_auto_network_default_v2_37(self):
        """Tests creating a server without specifying --nic using 2.37."""
        self.compute_client.api_version = api_versions.APIVersion('2.37')
        arglist = ['--image', 'image1', '--flavor', 'flavor1', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[], nics='auto', scheduler_hints={}, config_drive=None)
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)

    def _test_server_create_with_none_network(self, arglist):
        self.compute_client.api_version = api_versions.APIVersion('2.37')
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('nics', ['none']), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[], nics='none', scheduler_hints={}, config_drive=None)
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)

    def test_server_create_with_none_network_legacy(self):
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--nic', 'none', self.new_server.name]
        self._test_server_create_with_none_network(arglist)

    def test_server_create_with_none_network(self):
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--no-network', self.new_server.name]
        self._test_server_create_with_none_network(arglist)

    def test_server_create_with_none_network_pre_v237(self):
        self.compute_client.api_version = api_versions.APIVersion('2.36')
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--nic', 'none', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('nics', ['none']), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.37 or greater is required to support explicit auto-allocation of a network or to disable network allocation', str(exc))
        self.assertNotCalled(self.servers_mock.create)

    def test_server_create_with_conflict_network_options(self):
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--nic', 'none', '--nic', 'auto', '--nic', 'port-id=port1', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('nics', ['none', 'auto', {'net-id': '', 'port-id': 'port1', 'v4-fixed-ip': '', 'v6-fixed-ip': ''}]), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        get_endpoints = mock.Mock()
        get_endpoints.return_value = {'network': []}
        self.app.client_manager.auth_ref = mock.Mock()
        self.app.client_manager.auth_ref.service_catalog = mock.Mock()
        self.app.client_manager.auth_ref.service_catalog.get_endpoints = get_endpoints
        port_resource = mock.Mock(id='port1_uuid')
        self.network_client.find_port.return_value = port_resource
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertNotCalled(self.servers_mock.create)

    def test_server_create_with_invalid_network_options(self):
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--nic', 'abcdefgh', self.new_server.name]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, [])
        self.assertNotCalled(self.servers_mock.create)

    def test_server_create_with_invalid_network_key(self):
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--nic', 'abcdefgh=12324', self.new_server.name]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, [])
        self.assertNotCalled(self.servers_mock.create)

    def test_server_create_with_empty_network_key_value(self):
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--nic', 'net-id=', self.new_server.name]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, [])
        self.assertNotCalled(self.servers_mock.create)

    def test_server_create_with_only_network_key(self):
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--nic', 'net-id', self.new_server.name]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, [])
        self.assertNotCalled(self.servers_mock.create)

    @mock.patch.object(common_utils, 'wait_for_status', return_value=True)
    def test_server_create_with_wait_ok(self, mock_wait_for_status):
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--wait', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('config_drive', False), ('wait', True), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        mock_wait_for_status.assert_called_once_with(self.servers_mock.get, self.new_server.id, callback=mock.ANY)
        kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[], nics=[], scheduler_hints={}, config_drive=None)
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertTupleEqual(self.datalist(), data)

    @mock.patch.object(common_utils, 'wait_for_status', return_value=False)
    def test_server_create_with_wait_fails(self, mock_wait_for_status):
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--wait', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('config_drive', False), ('wait', True), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        mock_wait_for_status.assert_called_once_with(self.servers_mock.get, self.new_server.id, callback=mock.ANY)
        kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[], nics=[], scheduler_hints={}, config_drive=None)
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)

    @mock.patch('openstackclient.compute.v2.server.io.open')
    def test_server_create_userdata(self, mock_open):
        mock_file = mock.Mock(name='File')
        mock_open.return_value = mock_file
        mock_open.read.return_value = '#!/bin/sh'
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--user-data', 'userdata.sh', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('user_data', 'userdata.sh'), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        mock_open.assert_called_with('userdata.sh')
        mock_file.close.assert_called_with()
        kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=mock_file, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[], nics=[], scheduler_hints={}, config_drive=None)
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)

    def test_server_create_with_volume(self):
        arglist = ['--flavor', self.flavor.id, '--volume', self.volume.name, self.new_server.name]
        verifylist = [('flavor', self.flavor.id), ('volume', self.volume.name), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'meta': None, 'files': {}, 'reservation_id': None, 'min_count': 1, 'max_count': 1, 'security_groups': [], 'userdata': None, 'key_name': None, 'availability_zone': None, 'admin_pass': None, 'block_device_mapping_v2': [{'uuid': self.volume.id, 'boot_index': 0, 'source_type': 'volume', 'destination_type': 'volume'}], 'nics': [], 'scheduler_hints': {}, 'config_drive': None}
        self.servers_mock.create.assert_called_with(self.new_server.name, None, self.flavor, **kwargs)
        self.volumes_mock.get.assert_called_once_with(self.volume.name)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)

    def test_server_create_with_snapshot(self):
        arglist = ['--flavor', self.flavor.id, '--snapshot', self.snapshot.name, self.new_server.name]
        verifylist = [('flavor', self.flavor.id), ('snapshot', self.snapshot.name), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'meta': None, 'files': {}, 'reservation_id': None, 'min_count': 1, 'max_count': 1, 'security_groups': [], 'userdata': None, 'key_name': None, 'availability_zone': None, 'admin_pass': None, 'block_device_mapping_v2': [{'uuid': self.snapshot.id, 'boot_index': 0, 'source_type': 'snapshot', 'destination_type': 'volume', 'delete_on_termination': False}], 'nics': [], 'scheduler_hints': {}, 'config_drive': None}
        self.servers_mock.create.assert_called_with(self.new_server.name, None, self.flavor, **kwargs)
        self.snapshots_mock.get.assert_called_once_with(self.snapshot.name)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)

    def test_server_create_with_block_device(self):
        block_device = f'uuid={self.volume.id},source_type=volume,boot_index=0'
        arglist = ['--flavor', self.flavor.id, '--block-device', block_device, self.new_server.name]
        verifylist = [('image', None), ('flavor', self.flavor.id), ('block_devices', [{'uuid': self.volume.id, 'source_type': 'volume', 'boot_index': '0'}]), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'meta': None, 'files': {}, 'reservation_id': None, 'min_count': 1, 'max_count': 1, 'security_groups': [], 'userdata': None, 'key_name': None, 'availability_zone': None, 'admin_pass': None, 'block_device_mapping_v2': [{'uuid': self.volume.id, 'source_type': 'volume', 'destination_type': 'volume', 'boot_index': 0}], 'nics': [], 'scheduler_hints': {}, 'config_drive': None}
        self.servers_mock.create.assert_called_with(self.new_server.name, None, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)

    def test_server_create_with_block_device_full(self):
        self.compute_client.api_version = api_versions.APIVersion('2.67')
        block_device = f'uuid={self.volume.id},source_type=volume,destination_type=volume,disk_bus=ide,device_type=disk,device_name=sdb,guest_format=ext4,volume_size=64,volume_type=foo,boot_index=1,delete_on_termination=true,tag=foo'
        block_device_alt = f'uuid={self.volume_alt.id},source_type=volume'
        arglist = ['--image', 'image1', '--flavor', self.flavor.id, '--block-device', block_device, '--block-device', block_device_alt, self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', self.flavor.id), ('block_devices', [{'uuid': self.volume.id, 'source_type': 'volume', 'destination_type': 'volume', 'disk_bus': 'ide', 'device_type': 'disk', 'device_name': 'sdb', 'guest_format': 'ext4', 'volume_size': '64', 'volume_type': 'foo', 'boot_index': '1', 'delete_on_termination': 'true', 'tag': 'foo'}, {'uuid': self.volume_alt.id, 'source_type': 'volume'}]), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'meta': None, 'files': {}, 'reservation_id': None, 'min_count': 1, 'max_count': 1, 'security_groups': [], 'userdata': None, 'key_name': None, 'availability_zone': None, 'admin_pass': None, 'block_device_mapping_v2': [{'uuid': self.volume.id, 'source_type': 'volume', 'destination_type': 'volume', 'disk_bus': 'ide', 'device_name': 'sdb', 'volume_size': '64', 'guest_format': 'ext4', 'boot_index': 1, 'device_type': 'disk', 'delete_on_termination': True, 'tag': 'foo', 'volume_type': 'foo'}, {'uuid': self.volume_alt.id, 'source_type': 'volume', 'destination_type': 'volume'}], 'nics': 'auto', 'scheduler_hints': {}, 'config_drive': None}
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)

    def test_server_create_with_block_device_from_file(self):
        self.compute_client.api_version = api_versions.APIVersion('2.67')
        block_device = {'uuid': self.volume.id, 'source_type': 'volume', 'destination_type': 'volume', 'disk_bus': 'ide', 'device_type': 'disk', 'device_name': 'sdb', 'guest_format': 'ext4', 'volume_size': 64, 'volume_type': 'foo', 'boot_index': 1, 'delete_on_termination': True, 'tag': 'foo'}
        with tempfile.NamedTemporaryFile(mode='w+') as fp:
            json.dump(block_device, fp=fp)
            fp.flush()
            arglist = ['--image', 'image1', '--flavor', self.flavor.id, '--block-device', fp.name, self.new_server.name]
            verifylist = [('image', 'image1'), ('flavor', self.flavor.id), ('block_devices', [block_device]), ('server_name', self.new_server.name)]
            parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'meta': None, 'files': {}, 'reservation_id': None, 'min_count': 1, 'max_count': 1, 'security_groups': [], 'userdata': None, 'key_name': None, 'availability_zone': None, 'admin_pass': None, 'block_device_mapping_v2': [{'uuid': self.volume.id, 'source_type': 'volume', 'destination_type': 'volume', 'disk_bus': 'ide', 'device_name': 'sdb', 'volume_size': 64, 'guest_format': 'ext4', 'boot_index': 1, 'device_type': 'disk', 'delete_on_termination': True, 'tag': 'foo', 'volume_type': 'foo'}], 'nics': 'auto', 'scheduler_hints': {}, 'config_drive': None}
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)

    def test_server_create_with_block_device_invalid_boot_index(self):
        block_device = f'uuid={self.volume.name},source_type=volume,boot_index=foo'
        arglist = ['--image', 'image1', '--flavor', self.flavor.id, '--block-device', block_device, self.new_server.name]
        parsed_args = self.check_parser(self.cmd, arglist, [])
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('The boot_index key of --block-device ', str(ex))

    def test_server_create_with_block_device_invalid_source_type(self):
        block_device = f'uuid={self.volume.name},source_type=foo'
        arglist = ['--image', 'image1', '--flavor', self.flavor.id, '--block-device', block_device, self.new_server.name]
        parsed_args = self.check_parser(self.cmd, arglist, [])
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('The source_type key of --block-device ', str(ex))

    def test_server_create_with_block_device_invalid_destination_type(self):
        block_device = f'uuid={self.volume.name},destination_type=foo'
        arglist = ['--image', 'image1', '--flavor', self.flavor.id, '--block-device', block_device, self.new_server.name]
        parsed_args = self.check_parser(self.cmd, arglist, [])
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('The destination_type key of --block-device ', str(ex))

    def test_server_create_with_block_device_invalid_shutdown(self):
        block_device = f'uuid={self.volume.name},delete_on_termination=foo'
        arglist = ['--image', 'image1', '--flavor', self.flavor.id, '--block-device', block_device, self.new_server.name]
        parsed_args = self.check_parser(self.cmd, arglist, [])
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('The delete_on_termination key of --block-device ', str(ex))

    def test_server_create_with_block_device_tag_pre_v242(self):
        self.compute_client.api_version = api_versions.APIVersion('2.41')
        block_device = f'uuid={self.volume.name},tag=foo'
        arglist = ['--image', 'image1', '--flavor', self.flavor.id, '--block-device', block_device, self.new_server.name]
        parsed_args = self.check_parser(self.cmd, arglist, [])
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.42 or greater is required', str(ex))

    def test_server_create_with_block_device_volume_type_pre_v267(self):
        self.compute_client.api_version = api_versions.APIVersion('2.66')
        block_device = f'uuid={self.volume.name},volume_type=foo'
        arglist = ['--image', 'image1', '--flavor', self.flavor.id, '--block-device', block_device, self.new_server.name]
        parsed_args = self.check_parser(self.cmd, arglist, [])
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.67 or greater is required', str(ex))

    def test_server_create_with_block_device_mapping(self):
        arglist = ['--image', 'image1', '--flavor', self.flavor.id, '--block-device-mapping', 'vda=' + self.volume.name + ':::false', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', self.flavor.id), ('block_device_mapping', [{'device_name': 'vda', 'uuid': self.volume.name, 'source_type': 'volume', 'destination_type': 'volume', 'delete_on_termination': 'false'}]), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[{'device_name': 'vda', 'uuid': self.volume.id, 'destination_type': 'volume', 'source_type': 'volume', 'delete_on_termination': 'false'}], nics=[], scheduler_hints={}, config_drive=None)
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)

    def test_server_create_with_block_device_mapping_min_input(self):
        arglist = ['--image', 'image1', '--flavor', self.flavor.id, '--block-device-mapping', 'vdf=' + self.volume.name, self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', self.flavor.id), ('block_device_mapping', [{'device_name': 'vdf', 'uuid': self.volume.name, 'source_type': 'volume', 'destination_type': 'volume'}]), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[{'device_name': 'vdf', 'uuid': self.volume.id, 'destination_type': 'volume', 'source_type': 'volume'}], nics=[], scheduler_hints={}, config_drive=None)
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)

    def test_server_create_with_block_device_mapping_default_input(self):
        arglist = ['--image', 'image1', '--flavor', self.flavor.id, '--block-device-mapping', 'vdf=' + self.volume.name + ':::', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', self.flavor.id), ('block_device_mapping', [{'device_name': 'vdf', 'uuid': self.volume.name, 'source_type': 'volume', 'destination_type': 'volume'}]), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[{'device_name': 'vdf', 'uuid': self.volume.id, 'destination_type': 'volume', 'source_type': 'volume'}], nics=[], scheduler_hints={}, config_drive=None)
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)

    def test_server_create_with_block_device_mapping_full_input(self):
        arglist = ['--image', 'image1', '--flavor', self.flavor.id, '--block-device-mapping', 'vde=' + self.volume.name + ':volume:3:true', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', self.flavor.id), ('block_device_mapping', [{'device_name': 'vde', 'uuid': self.volume.name, 'source_type': 'volume', 'destination_type': 'volume', 'volume_size': '3', 'delete_on_termination': 'true'}]), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[{'device_name': 'vde', 'uuid': self.volume.id, 'destination_type': 'volume', 'source_type': 'volume', 'delete_on_termination': 'true', 'volume_size': '3'}], nics=[], scheduler_hints={}, config_drive=None)
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)

    def test_server_create_with_block_device_mapping_snapshot(self):
        arglist = ['--image', 'image1', '--flavor', self.flavor.id, '--block-device-mapping', 'vds=' + self.volume.name + ':snapshot:5:true', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', self.flavor.id), ('block_device_mapping', [{'device_name': 'vds', 'uuid': self.volume.name, 'source_type': 'snapshot', 'volume_size': '5', 'destination_type': 'volume', 'delete_on_termination': 'true'}]), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[{'device_name': 'vds', 'uuid': self.snapshot.id, 'destination_type': 'volume', 'source_type': 'snapshot', 'delete_on_termination': 'true', 'volume_size': '5'}], nics=[], scheduler_hints={}, config_drive=None)
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)

    def test_server_create_with_block_device_mapping_multiple(self):
        arglist = ['--image', 'image1', '--flavor', self.flavor.id, '--block-device-mapping', 'vdb=' + self.volume.name + ':::false', '--block-device-mapping', 'vdc=' + self.volume.name + ':::true', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', self.flavor.id), ('block_device_mapping', [{'device_name': 'vdb', 'uuid': self.volume.name, 'source_type': 'volume', 'destination_type': 'volume', 'delete_on_termination': 'false'}, {'device_name': 'vdc', 'uuid': self.volume.name, 'source_type': 'volume', 'destination_type': 'volume', 'delete_on_termination': 'true'}]), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[{'device_name': 'vdb', 'uuid': self.volume.id, 'destination_type': 'volume', 'source_type': 'volume', 'delete_on_termination': 'false'}, {'device_name': 'vdc', 'uuid': self.volume.id, 'destination_type': 'volume', 'source_type': 'volume', 'delete_on_termination': 'true'}], nics=[], scheduler_hints={}, config_drive=None)
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)

    def test_server_create_with_block_device_mapping_invalid_format(self):
        arglist = ['--image', 'image1', '--flavor', self.flavor.id, '--block-device-mapping', 'not_contain_equal_sign', self.new_server.name]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, [])
        arglist = ['--image', 'image1', '--flavor', self.flavor.id, '--block-device-mapping', '=uuid:::true', self.new_server.name]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, [])

    def test_server_create_with_block_device_mapping_no_uuid(self):
        arglist = ['--image', 'image1', '--flavor', self.flavor.id, '--block-device-mapping', 'vdb=', self.new_server.name]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, [])

    def test_server_create_volume_boot_from_volume_conflict(self):
        arglist = ['--flavor', self.flavor.id, '--volume', 'volume1', '--boot-from-volume', '1', self.new_server.name]
        verifylist = [('flavor', self.flavor.id), ('volume', 'volume1'), ('boot_from_volume', 1), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--volume is not allowed with --boot-from-volume', str(ex))

    def test_server_create_boot_from_volume_no_image(self):
        arglist = ['--flavor', self.flavor.id, '--boot-from-volume', '1', self.new_server.name]
        verifylist = [('flavor', self.flavor.id), ('boot_from_volume', 1), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('An image (--image or --image-property) is required to support --boot-from-volume option', str(ex))

    def test_server_create_image_property(self):
        arglist = ['--image-property', 'hypervisor_type=qemu', '--flavor', 'flavor1', self.new_server.name]
        verifylist = [('image_properties', {'hypervisor_type': 'qemu'}), ('flavor', 'flavor1'), ('config_drive', False), ('server_name', self.new_server.name)]
        image_info = {'hypervisor_type': 'qemu'}
        _image = image_fakes.create_one_image(image_info)
        self.image_client.images.return_value = [_image]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = dict(files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[], nics=[], meta=None, scheduler_hints={}, config_drive=None)
        self.servers_mock.create.assert_called_with(self.new_server.name, _image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)

    def test_server_create_image_property_multi(self):
        arglist = ['--image-property', 'hypervisor_type=qemu', '--image-property', 'hw_disk_bus=ide', '--flavor', 'flavor1', self.new_server.name]
        verifylist = [('image_properties', {'hypervisor_type': 'qemu', 'hw_disk_bus': 'ide'}), ('flavor', 'flavor1'), ('config_drive', False), ('server_name', self.new_server.name)]
        image_info = {'hypervisor_type': 'qemu', 'hw_disk_bus': 'ide'}
        _image = image_fakes.create_one_image(image_info)
        self.image_client.images.return_value = [_image]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = dict(files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[], nics=[], meta=None, scheduler_hints={}, config_drive=None)
        self.servers_mock.create.assert_called_with(self.new_server.name, _image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)

    def test_server_create_image_property_missed(self):
        arglist = ['--image-property', 'hypervisor_type=qemu', '--image-property', 'hw_disk_bus=virtio', '--flavor', 'flavor1', self.new_server.name]
        verifylist = [('image_properties', {'hypervisor_type': 'qemu', 'hw_disk_bus': 'virtio'}), ('flavor', 'flavor1'), ('config_drive', False), ('server_name', self.new_server.name)]
        image_info = {'hypervisor_type': 'qemu', 'hw_disk_bus': 'ide'}
        _image = image_fakes.create_one_image(image_info)
        self.image_client.images.return_value = [_image]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_server_create_image_property_with_image_list(self):
        arglist = ['--image-property', 'owner_specified.openstack.object=image/cirros', '--flavor', 'flavor1', self.new_server.name]
        verifylist = [('image_properties', {'owner_specified.openstack.object': 'image/cirros'}), ('flavor', 'flavor1'), ('server_name', self.new_server.name)]
        image_info = {'properties': {'owner_specified.openstack.object': 'image/cirros'}}
        target_image = image_fakes.create_one_image(image_info)
        another_image = image_fakes.create_one_image({})
        self.image_client.images.return_value = [target_image, another_image]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = dict(files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[], nics=[], meta=None, scheduler_hints={}, config_drive=None)
        self.servers_mock.create.assert_called_with(self.new_server.name, target_image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)

    def test_server_create_no_boot_device(self):
        block_device = f'uuid={self.volume.id},source_type=volume,boot_index=1'
        arglist = ['--block-device', block_device, '--flavor', self.flavor.id, self.new_server.name]
        verifylist = [('image', None), ('flavor', self.flavor.id), ('block_devices', [{'uuid': self.volume.id, 'source_type': 'volume', 'boot_index': '1'}]), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('An image (--image, --image-property) or bootable volume (--volume, --snapshot, --block-device) is required', str(exc))

    def test_server_create_with_swap(self):
        arglist = ['--image', 'image1', '--flavor', self.flavor.id, '--swap', '1024', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', self.flavor.id), ('swap', 1024), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'meta': None, 'files': {}, 'reservation_id': None, 'min_count': 1, 'max_count': 1, 'security_groups': [], 'userdata': None, 'key_name': None, 'availability_zone': None, 'admin_pass': None, 'block_device_mapping_v2': [{'boot_index': -1, 'source_type': 'blank', 'destination_type': 'local', 'guest_format': 'swap', 'volume_size': 1024, 'delete_on_termination': True}], 'nics': [], 'scheduler_hints': {}, 'config_drive': None}
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)

    def test_server_create_with_ephemeral(self):
        arglist = ['--image', 'image1', '--flavor', self.flavor.id, '--ephemeral', 'size=1024,format=ext4', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', self.flavor.id), ('ephemerals', [{'size': '1024', 'format': 'ext4'}]), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'meta': None, 'files': {}, 'reservation_id': None, 'min_count': 1, 'max_count': 1, 'security_groups': [], 'userdata': None, 'key_name': None, 'availability_zone': None, 'admin_pass': None, 'block_device_mapping_v2': [{'boot_index': -1, 'source_type': 'blank', 'destination_type': 'local', 'guest_format': 'ext4', 'volume_size': '1024', 'delete_on_termination': True}], 'nics': [], 'scheduler_hints': {}, 'config_drive': None}
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)

    def test_server_create_with_ephemeral_missing_key(self):
        arglist = ['--image', 'image1', '--flavor', self.flavor.id, '--ephemeral', 'format=ext3', self.new_server.name]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, [])

    def test_server_create_with_ephemeral_invalid_key(self):
        arglist = ['--image', 'image1', '--flavor', self.flavor.id, '--ephemeral', 'size=1024,foo=bar', self.new_server.name]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, [])

    def test_server_create_invalid_hint(self):
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--hint', 'a0cf03a5-d921-4877-bb5c-86d26cf818e1', self.new_server.name]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, [])
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--hint', '=a0cf03a5-d921-4877-bb5c-86d26cf818e1', self.new_server.name]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, [])

    def test_server_create_with_description_api_newer(self):
        self.compute_client.api_version = 2.19
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--description', 'description1', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('description', 'description1'), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(api_versions, 'APIVersion', return_value=2.19):
            columns, data = self.cmd.take_action(parsed_args)
        kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[], nics='auto', scheduler_hints={}, config_drive=None, description='description1')
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)
        self.assertFalse(self.image_client.images.called)
        self.assertFalse(self.flavors_mock.called)

    def test_server_create_with_description_api_older(self):
        self.compute_client.api_version = 2.18
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--description', 'description1', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('description', 'description1'), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(api_versions, 'APIVersion', return_value=2.19):
            self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_server_create_with_tag(self):
        self.compute_client.api_version = api_versions.APIVersion('2.52')
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--tag', 'tag1', '--tag', 'tag2', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('tags', ['tag1', 'tag2']), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'meta': None, 'files': {}, 'reservation_id': None, 'min_count': 1, 'max_count': 1, 'security_groups': [], 'userdata': None, 'key_name': None, 'availability_zone': None, 'block_device_mapping_v2': [], 'admin_pass': None, 'nics': 'auto', 'scheduler_hints': {}, 'config_drive': None, 'tags': ['tag1', 'tag2']}
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)
        self.assertFalse(self.image_client.images.called)
        self.assertFalse(self.flavors_mock.called)

    def test_server_create_with_tag_pre_v252(self):
        self.compute_client.api_version = api_versions.APIVersion('2.51')
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--tag', 'tag1', '--tag', 'tag2', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('tags', ['tag1', 'tag2']), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.52 or greater is required', str(ex))

    def test_server_create_with_host_v274(self):
        self.compute_client.api_version = 2.74
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--host', 'host1', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('host', 'host1'), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(api_versions, 'APIVersion', return_value=2.74):
            columns, data = self.cmd.take_action(parsed_args)
        kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[], nics='auto', scheduler_hints={}, config_drive=None, host='host1')
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)
        self.assertFalse(self.image_client.images.called)
        self.assertFalse(self.flavors_mock.called)

    def test_server_create_with_host_pre_v274(self):
        self.compute_client.api_version = 2.73
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--host', 'host1', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('host', 'host1'), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(api_versions, 'APIVersion', return_value=2.74):
            self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_server_create_with_hypervisor_hostname_v274(self):
        self.compute_client.api_version = 2.74
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--hypervisor-hostname', 'node1', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('hypervisor_hostname', 'node1'), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(api_versions, 'APIVersion', return_value=2.74):
            columns, data = self.cmd.take_action(parsed_args)
        kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[], nics='auto', scheduler_hints={}, config_drive=None, hypervisor_hostname='node1')
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)
        self.assertFalse(self.image_client.images.called)
        self.assertFalse(self.flavors_mock.called)

    def test_server_create_with_hypervisor_hostname_pre_v274(self):
        self.compute_client.api_version = 2.73
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--hypervisor-hostname', 'node1', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('hypervisor_hostname', 'node1'), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(api_versions, 'APIVersion', return_value=2.74):
            self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_server_create_with_host_and_hypervisor_hostname_v274(self):
        self.compute_client.api_version = 2.74
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--host', 'host1', '--hypervisor-hostname', 'node1', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('host', 'host1'), ('hypervisor_hostname', 'node1'), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(api_versions, 'APIVersion', return_value=2.74):
            columns, data = self.cmd.take_action(parsed_args)
        kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[], nics='auto', scheduler_hints={}, config_drive=None, host='host1', hypervisor_hostname='node1')
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)
        self.assertFalse(self.image_client.images.called)
        self.assertFalse(self.flavors_mock.called)

    def test_server_create_with_hostname_v290(self):
        self.compute_client.api_version = api_versions.APIVersion('2.90')
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--hostname', 'hostname', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('hostname', 'hostname'), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[], nics='auto', scheduler_hints={}, config_drive=None, hostname='hostname')
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)
        self.assertFalse(self.image_client.images.called)
        self.assertFalse(self.flavors_mock.called)

    def test_server_create_with_hostname_pre_v290(self):
        self.compute_client.api_version = api_versions.APIVersion('2.89')
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--hostname', 'hostname', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('hostname', 'hostname'), ('config_drive', False), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_server_create_with_trusted_image_cert(self):
        self.compute_client.api_version = api_versions.APIVersion('2.63')
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--trusted-image-cert', 'foo', '--trusted-image-cert', 'bar', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('config_drive', False), ('trusted_image_certs', ['foo', 'bar']), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[], nics='auto', scheduler_hints={}, config_drive=None, trusted_image_certificates=['foo', 'bar'])
        self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist(), data)
        self.assertFalse(self.image_client.images.called)
        self.assertFalse(self.flavors_mock.called)

    def test_server_create_with_trusted_image_cert_prev263(self):
        self.compute_client.api_version = api_versions.APIVersion('2.62')
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--trusted-image-cert', 'foo', '--trusted-image-cert', 'bar', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('config_drive', False), ('trusted_image_certs', ['foo', 'bar']), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_server_create_with_trusted_image_cert_from_volume(self):
        self.compute_client.api_version = api_versions.APIVersion('2.63')
        arglist = ['--volume', 'volume1', '--flavor', 'flavor1', '--trusted-image-cert', 'foo', '--trusted-image-cert', 'bar', self.new_server.name]
        verifylist = [('volume', 'volume1'), ('flavor', 'flavor1'), ('config_drive', False), ('trusted_image_certs', ['foo', 'bar']), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_server_create_with_trusted_image_cert_from_snapshot(self):
        self.compute_client.api_version = api_versions.APIVersion('2.63')
        arglist = ['--snapshot', 'snapshot1', '--flavor', 'flavor1', '--trusted-image-cert', 'foo', '--trusted-image-cert', 'bar', self.new_server.name]
        verifylist = [('snapshot', 'snapshot1'), ('flavor', 'flavor1'), ('config_drive', False), ('trusted_image_certs', ['foo', 'bar']), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_server_create_with_trusted_image_cert_boot_from_volume(self):
        self.compute_client.api_version = api_versions.APIVersion('2.63')
        arglist = ['--image', 'image1', '--flavor', 'flavor1', '--boot-from-volume', '1', '--trusted-image-cert', 'foo', '--trusted-image-cert', 'bar', self.new_server.name]
        verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('boot_from_volume', 1), ('config_drive', False), ('trusted_image_certs', ['foo', 'bar']), ('server_name', self.new_server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)