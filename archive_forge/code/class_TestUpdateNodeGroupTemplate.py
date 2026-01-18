from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import node_group_templates as api_ngt
from saharaclient.osc.v2 import node_group_templates as osc_ngt
from saharaclient.tests.unit.osc.v1 import fakes
class TestUpdateNodeGroupTemplate(TestNodeGroupTemplates):

    def setUp(self):
        super(TestUpdateNodeGroupTemplate, self).setUp()
        self.ngt_mock.find_unique.return_value = api_ngt.NodeGroupTemplate(None, NGT_INFO)
        self.ngt_mock.update.return_value = api_ngt.NodeGroupTemplate(None, NGT_INFO)
        self.fl_mock = self.app.client_manager.compute.flavors
        self.fl_mock.get.return_value = mock.Mock(id='flavor_id')
        self.fl_mock.reset_mock()
        self.cmd = osc_ngt.UpdateNodeGroupTemplate(self.app, None)

    def test_ngt_update_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_ngt_update_nothing_updated(self):
        arglist = ['template']
        verifylist = [('node_group_template', 'template')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.ngt_mock.update.assert_called_once_with('ng_id')

    def test_ngt_update_all_options(self):
        arglist = ['template', '--name', 'template', '--plugin', 'fake', '--plugin-version', '0.1', '--processes', 'namenode', 'tasktracker', '--security-groups', 'secgr', '--auto-security-group-enable', '--availability-zone', 'av_zone', '--flavor', 'flavor_id', '--floating-ip-pool', 'floating_pool', '--volumes-per-node', '2', '--volumes-size', '2', '--volumes-type', 'type', '--volumes-availability-zone', 'vavzone', '--volumes-mount-prefix', '/volume/asd', '--volumes-locality-enable', '--description', 'descr', '--autoconfig-enable', '--proxy-gateway-enable', '--public', '--protected', '--boot-from-volume-enable', '--boot-volume-type', 'volume2', '--boot-volume-availability-zone', 'ceph', '--boot-volume-local-to-instance-enable']
        verifylist = [('node_group_template', 'template'), ('name', 'template'), ('plugin', 'fake'), ('plugin_version', '0.1'), ('processes', ['namenode', 'tasktracker']), ('security_groups', ['secgr']), ('use_auto_security_group', True), ('availability_zone', 'av_zone'), ('flavor', 'flavor_id'), ('floating_ip_pool', 'floating_pool'), ('volumes_per_node', 2), ('volumes_size', 2), ('volumes_type', 'type'), ('volumes_availability_zone', 'vavzone'), ('volumes_mount_prefix', '/volume/asd'), ('volume_locality', True), ('description', 'descr'), ('use_autoconfig', True), ('is_proxy_gateway', True), ('is_public', True), ('is_protected', True), ('boot_from_volume', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.ngt_mock.update.assert_called_once_with('ng_id', auto_security_group=True, availability_zone='av_zone', description='descr', flavor_id='flavor_id', floating_ip_pool='floating_pool', plugin_version='0.1', is_protected=True, is_proxy_gateway=True, is_public=True, name='template', node_processes=['namenode', 'tasktracker'], plugin_name='fake', security_groups=['secgr'], use_autoconfig=True, volume_local_to_instance=True, volume_type='type', volumes_availability_zone='vavzone', volumes_per_node=2, volumes_size=2, volume_mount_prefix='/volume/asd', boot_from_volume=True, boot_volume_type='volume2', boot_volume_availability_zone='ceph', boot_volume_local_to_instance=True)
        expected_columns = ('Auto security group', 'Availability zone', 'Boot from volume', 'Description', 'Flavor id', 'Floating ip pool', 'Id', 'Is default', 'Is protected', 'Is proxy gateway', 'Is public', 'Name', 'Node processes', 'Plugin name', 'Plugin version', 'Security groups', 'Use autoconfig', 'Volume local to instance', 'Volume mount prefix', 'Volume type', 'Volumes availability zone', 'Volumes per node', 'Volumes size')
        self.assertEqual(expected_columns, columns)
        expected_data = (True, 'av_zone', False, 'description', 'flavor_id', 'floating_pool', 'ng_id', False, False, False, True, 'template', 'namenode, tasktracker', 'fake', '0.1', None, True, False, '/volumes/disk', None, None, 2, 2)
        self.assertEqual(expected_data, data)

    def test_ngt_update_private_unprotected(self):
        arglist = ['template', '--private', '--unprotected']
        verifylist = [('node_group_template', 'template'), ('is_public', False), ('is_protected', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.ngt_mock.update.assert_called_once_with('ng_id', is_protected=False, is_public=False)