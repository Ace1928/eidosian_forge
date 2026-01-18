from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import node_group_templates as api_ngt
from saharaclient.osc.v2 import node_group_templates as osc_ngt
from saharaclient.tests.unit.osc.v1 import fakes
class TestDeleteNodeGroupTemplate(TestNodeGroupTemplates):

    def setUp(self):
        super(TestDeleteNodeGroupTemplate, self).setUp()
        self.ngt_mock.find_unique.return_value = api_ngt.NodeGroupTemplate(None, NGT_INFO)
        self.cmd = osc_ngt.DeleteNodeGroupTemplate(self.app, None)

    def test_ngt_delete(self):
        arglist = ['template']
        verifylist = [('node_group_template', ['template'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.ngt_mock.delete.assert_called_once_with('ng_id')