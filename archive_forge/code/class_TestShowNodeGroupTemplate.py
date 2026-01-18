from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import node_group_templates as api_ngt
from saharaclient.osc.v2 import node_group_templates as osc_ngt
from saharaclient.tests.unit.osc.v1 import fakes
class TestShowNodeGroupTemplate(TestNodeGroupTemplates):

    def setUp(self):
        super(TestShowNodeGroupTemplate, self).setUp()
        self.ngt_mock.find_unique.return_value = api_ngt.NodeGroupTemplate(None, NGT_INFO)
        self.cmd = osc_ngt.ShowNodeGroupTemplate(self.app, None)

    def test_ngt_show(self):
        arglist = ['template']
        verifylist = [('node_group_template', 'template')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.ngt_mock.find_unique.assert_called_once_with(name='template')
        expected_columns = ('Auto security group', 'Availability zone', 'Boot from volume', 'Description', 'Flavor id', 'Floating ip pool', 'Id', 'Is default', 'Is protected', 'Is proxy gateway', 'Is public', 'Name', 'Node processes', 'Plugin name', 'Plugin version', 'Security groups', 'Use autoconfig', 'Volume local to instance', 'Volume mount prefix', 'Volume type', 'Volumes availability zone', 'Volumes per node', 'Volumes size')
        self.assertEqual(expected_columns, columns)
        expected_data = (True, 'av_zone', False, 'description', 'flavor_id', 'floating_pool', 'ng_id', False, False, False, True, 'template', 'namenode, tasktracker', 'fake', '0.1', None, True, False, '/volumes/disk', None, None, 2, 2)
        self.assertEqual(expected_data, data)