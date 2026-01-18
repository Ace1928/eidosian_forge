from unittest import mock
from heat.common import grouputils
from heat.common import identifier
from heat.common import template_format
from heat.rpc import client as rpc_client
from heat.tests import common
from heat.tests import utils
class GroupInspectorTest(common.HeatTestCase):
    resources = [{'updated_time': '2018-01-01T12:00', 'creation_time': '2018-01-01T02:00', 'resource_name': 'A', 'physical_resource_id': 'a', 'resource_action': 'UPDATE', 'resource_status': 'COMPLETE', 'resource_status_reason': 'resource changed', 'resource_type': 'OS::Heat::Test', 'resource_id': 'aaaaaaaa', 'stack_identity': 'bar', 'stack_name': 'nested_test', 'required_by': [], 'parent_resource': 'stack_resource'}, {'updated_time': '2018-01-01T10:00', 'creation_time': '2018-01-01T03:00', 'resource_name': 'E', 'physical_resource_id': 'e', 'resource_action': 'UPDATE', 'resource_status': 'FAILED', 'resource_status_reason': 'reasons', 'resource_type': 'OS::Heat::Test', 'resource_id': 'eeeeeeee', 'stack_identity': 'bar', 'stack_name': 'nested_test', 'required_by': [], 'parent_resource': 'stack_resource'}, {'updated_time': '2018-01-01T11:00', 'creation_time': '2018-01-01T03:00', 'resource_name': 'B', 'physical_resource_id': 'b', 'resource_action': 'UPDATE', 'resource_status': 'FAILED', 'resource_status_reason': 'reasons', 'resource_type': 'OS::Heat::Test', 'resource_id': 'bbbbbbbb', 'stack_identity': 'bar', 'stack_name': 'nested_test', 'required_by': [], 'parent_resource': 'stack_resource'}, {'updated_time': '2018-01-01T13:00', 'creation_time': '2018-01-01T01:00', 'resource_name': 'C', 'physical_resource_id': 'c', 'resource_action': 'UPDATE', 'resource_status': 'COMPLETE', 'resource_status_reason': 'resource changed', 'resource_type': 'OS::Heat::Test', 'resource_id': 'cccccccc', 'stack_identity': 'bar', 'stack_name': 'nested_test', 'required_by': [], 'parent_resource': 'stack_resource'}, {'updated_time': '2018-01-01T04:00', 'creation_time': '2018-01-01T04:00', 'resource_name': 'F', 'physical_resource_id': 'f', 'resource_action': 'CREATE', 'resource_status': 'COMPLETE', 'resource_status_reason': 'resource changed', 'resource_type': 'OS::Heat::Test', 'resource_id': 'ffffffff', 'stack_identity': 'bar', 'stack_name': 'nested_test', 'required_by': [], 'parent_resource': 'stack_resource'}, {'updated_time': '2018-01-01T04:00', 'creation_time': '2018-01-01T04:00', 'resource_name': 'D', 'physical_resource_id': 'd', 'resource_action': 'CREATE', 'resource_status': 'COMPLETE', 'resource_status_reason': 'resource changed', 'resource_type': 'OS::Heat::Test', 'resource_id': 'dddddddd', 'stack_identity': 'bar', 'stack_name': 'nested_test', 'required_by': [], 'parent_resource': 'stack_resource'}]
    template = {'heat_template_version': 'newton', 'resources': {'A': {'type': 'OS::Heat::TestResource'}}}

    def setUp(self):
        super(GroupInspectorTest, self).setUp()
        self.ctx = mock.Mock()
        self.rpc_client = mock.Mock(spec=rpc_client.EngineClient)
        self.identity = identifier.HeatIdentifier('foo', 'nested_test', 'bar')
        self.list_rsrcs = self.rpc_client.list_stack_resources
        self.get_tmpl = self.rpc_client.get_template
        self.insp = grouputils.GroupInspector(self.ctx, self.rpc_client, self.identity)

    def test_no_identity(self):
        self.insp = grouputils.GroupInspector(self.ctx, self.rpc_client, None)
        self.assertEqual(0, self.insp.size(include_failed=True))
        self.assertEqual([], list(self.insp.member_names(include_failed=True)))
        self.assertIsNone(self.insp.template())
        self.list_rsrcs.assert_not_called()
        self.get_tmpl.assert_not_called()

    def test_size_include_failed(self):
        self.list_rsrcs.return_value = self.resources
        self.assertEqual(6, self.insp.size(include_failed=True))
        self.list_rsrcs.assert_called_once_with(self.ctx, dict(self.identity))

    def test_size_exclude_failed(self):
        self.list_rsrcs.return_value = self.resources
        self.assertEqual(4, self.insp.size(include_failed=False))
        self.list_rsrcs.assert_called_once_with(self.ctx, dict(self.identity))

    def test_member_names_include_failed(self):
        self.list_rsrcs.return_value = self.resources
        self.assertEqual(['B', 'E', 'C', 'A', 'D', 'F'], list(self.insp.member_names(include_failed=True)))
        self.list_rsrcs.assert_called_once_with(self.ctx, dict(self.identity))

    def test_member_names_exclude_failed(self):
        self.list_rsrcs.return_value = self.resources
        self.assertEqual(['C', 'A', 'D', 'F'], list(self.insp.member_names(include_failed=False)))
        self.list_rsrcs.assert_called_once_with(self.ctx, dict(self.identity))

    def test_list_rsrc_caching(self):
        self.list_rsrcs.return_value = self.resources
        self.insp.size(include_failed=False)
        list(self.insp.member_names(include_failed=True))
        self.insp.size(include_failed=True)
        list(self.insp.member_names(include_failed=False))
        self.list_rsrcs.assert_called_once_with(self.ctx, dict(self.identity))
        self.get_tmpl.assert_not_called()

    def test_get_template(self):
        self.get_tmpl.return_value = self.template
        tmpl = self.insp.template()
        self.assertEqual(self.template, tmpl.t)
        self.get_tmpl.assert_called_once_with(self.ctx, dict(self.identity))

    def test_get_tmpl_caching(self):
        self.get_tmpl.return_value = self.template
        self.insp.template()
        self.insp.template()
        self.get_tmpl.assert_called_once_with(self.ctx, dict(self.identity))
        self.list_rsrcs.assert_not_called()