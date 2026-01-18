from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import node_group_templates as api_ngt
from saharaclient.osc.v2 import node_group_templates as osc_ngt
from saharaclient.tests.unit.osc.v1 import fakes
class TestCreateNodeGroupTemplate(TestNodeGroupTemplates):

    def setUp(self):
        super(TestCreateNodeGroupTemplate, self).setUp()
        self.ngt_mock.create.return_value = api_ngt.NodeGroupTemplate(None, NGT_INFO)
        self.fl_mock = self.app.client_manager.compute.flavors
        self.fl_mock.get.return_value = mock.Mock(id='flavor_id')
        self.fl_mock.reset_mock()
        self.cmd = osc_ngt.CreateNodeGroupTemplate(self.app, None)

    def test_ngt_create_minimum_options(self):
        arglist = ['--name', 'template', '--plugin', 'fake', '--plugin-version', '0.1', '--processes', 'namenode', 'tasktracker', '--flavor', 'flavor_id']
        verifylist = [('name', 'template'), ('plugin', 'fake'), ('plugin_version', '0.1'), ('flavor', 'flavor_id'), ('processes', ['namenode', 'tasktracker'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.ngt_mock.create.assert_called_once_with(auto_security_group=False, availability_zone=None, description=None, flavor_id='flavor_id', floating_ip_pool=None, plugin_version='0.1', is_protected=False, is_proxy_gateway=False, is_public=False, name='template', node_processes=['namenode', 'tasktracker'], plugin_name='fake', security_groups=None, use_autoconfig=False, volume_local_to_instance=False, volume_type=None, volumes_availability_zone=None, volumes_per_node=None, volumes_size=None, shares=None, node_configs=None, volume_mount_prefix=None, boot_from_volume=False, boot_volume_type=None, boot_volume_availability_zone=None, boot_volume_local_to_instance=False)

    def test_ngt_create_all_options(self):
        arglist = ['--name', 'template', '--plugin', 'fake', '--plugin-version', '0.1', '--processes', 'namenode', 'tasktracker', '--security-groups', 'secgr', '--auto-security-group', '--availability-zone', 'av_zone', '--flavor', 'flavor_id', '--floating-ip-pool', 'floating_pool', '--volumes-per-node', '2', '--volumes-size', '2', '--volumes-type', 'type', '--volumes-availability-zone', 'vavzone', '--volumes-mount-prefix', '/volume/asd', '--volumes-locality', '--description', 'descr', '--autoconfig', '--proxy-gateway', '--public', '--protected', '--boot-from-volume', '--boot-volume-type', 'volume2', '--boot-volume-availability-zone', 'ceph', '--boot-volume-local-to-instance']
        verifylist = [('name', 'template'), ('plugin', 'fake'), ('plugin_version', '0.1'), ('processes', ['namenode', 'tasktracker']), ('security_groups', ['secgr']), ('auto_security_group', True), ('availability_zone', 'av_zone'), ('flavor', 'flavor_id'), ('floating_ip_pool', 'floating_pool'), ('volumes_per_node', 2), ('volumes_size', 2), ('volumes_type', 'type'), ('volumes_availability_zone', 'vavzone'), ('volumes_mount_prefix', '/volume/asd'), ('volumes_locality', True), ('description', 'descr'), ('autoconfig', True), ('proxy_gateway', True), ('public', True), ('protected', True), ('boot_from_volume', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.ngt_mock.create.assert_called_once_with(auto_security_group=True, availability_zone='av_zone', description='descr', flavor_id='flavor_id', floating_ip_pool='floating_pool', plugin_version='0.1', is_protected=True, is_proxy_gateway=True, is_public=True, name='template', node_processes=['namenode', 'tasktracker'], plugin_name='fake', security_groups=['secgr'], use_autoconfig=True, volume_local_to_instance=True, volume_type='type', volumes_availability_zone='vavzone', volumes_per_node=2, volumes_size=2, shares=None, node_configs=None, volume_mount_prefix='/volume/asd', boot_from_volume=True, boot_volume_type='volume2', boot_volume_availability_zone='ceph', boot_volume_local_to_instance=True)
        expected_columns = ('Auto security group', 'Availability zone', 'Boot from volume', 'Description', 'Flavor id', 'Floating ip pool', 'Id', 'Is default', 'Is protected', 'Is proxy gateway', 'Is public', 'Name', 'Node processes', 'Plugin name', 'Plugin version', 'Security groups', 'Use autoconfig', 'Volume local to instance', 'Volume mount prefix', 'Volume type', 'Volumes availability zone', 'Volumes per node', 'Volumes size')
        self.assertEqual(expected_columns, columns)
        expected_data = (True, 'av_zone', False, 'description', 'flavor_id', 'floating_pool', 'ng_id', False, False, False, True, 'template', 'namenode, tasktracker', 'fake', '0.1', None, True, False, '/volumes/disk', None, None, 2, 2)
        self.assertEqual(expected_data, data)