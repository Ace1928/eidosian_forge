from unittest import mock
from heat.common import grouputils
from heat.common import identifier
from heat.common import template_format
from heat.rpc import client as rpc_client
from heat.tests import common
from heat.tests import utils
class GroupUtilsTest(common.HeatTestCase):

    def test_non_nested_resource(self):
        group = mock.Mock()
        group.nested_identifier.return_value = None
        group.nested.return_value = None
        self.assertEqual(0, grouputils.get_size(group))
        self.assertEqual([], grouputils.get_members(group))
        self.assertEqual([], grouputils.get_member_refids(group))
        self.assertEqual([], grouputils.get_member_names(group))

    def test_normal_group(self):
        group = mock.Mock()
        t = template_format.parse(nested_stack)
        stack = utils.parse_stack(t)
        group.nested.return_value = stack
        members = [r for r in stack.values()]
        expected = sorted(members, key=lambda r: (r.created_time, r.name))
        actual = grouputils.get_members(group)
        self.assertEqual(expected, actual)
        actual_ids = grouputils.get_member_refids(group)
        self.assertEqual(['ID-r0', 'ID-r1'], actual_ids)

    def test_group_with_failed_members(self):
        group = mock.Mock()
        t = template_format.parse(nested_stack)
        stack = utils.parse_stack(t)
        self.patchobject(group, 'nested', return_value=stack)
        rsrc_err = stack.resources['r0']
        rsrc_err.status = rsrc_err.FAILED
        rsrc_ok = stack.resources['r1']
        self.assertEqual([rsrc_ok], grouputils.get_members(group))
        self.assertEqual(['ID-r1'], grouputils.get_member_refids(group))