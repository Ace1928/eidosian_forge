from unittest import mock
from mistralclient.api.v2 import members
from mistralclient.commands.v2 import members as member_cmd
from mistralclient.tests.unit import base
class TestCLIWorkflowMembers(base.BaseCommandTest):

    def test_create(self):
        self.client.members.create.return_value = MEMBER
        result = self.call(member_cmd.Create, app_args=[MEMBER_DICT['resource_id'], MEMBER_DICT['resource_type'], MEMBER_DICT['member_id']])
        self.assertEqual(('456', 'workflow', '1111', '2222', 'pending', '1', '1'), result[1])

    def test_update(self):
        self.client.members.update.return_value = MEMBER
        result = self.call(member_cmd.Update, app_args=[MEMBER_DICT['resource_id'], MEMBER_DICT['resource_type'], '-m', MEMBER_DICT['member_id']])
        self.assertEqual(('456', 'workflow', '1111', '2222', 'pending', '1', '1'), result[1])

    def test_list(self):
        self.client.members.list.return_value = [MEMBER]
        result = self.call(member_cmd.List, app_args=[MEMBER_DICT['resource_id'], MEMBER_DICT['resource_type']])
        self.assertListEqual([('456', 'workflow', '1111', '2222', 'pending', '1', '1')], result[1])

    def test_get(self):
        self.client.members.get.return_value = MEMBER
        result = self.call(member_cmd.Get, app_args=[MEMBER_DICT['resource_id'], MEMBER_DICT['resource_type'], '-m', MEMBER_DICT['member_id']])
        self.assertEqual(('456', 'workflow', '1111', '2222', 'pending', '1', '1'), result[1])

    def test_delete(self):
        self.call(member_cmd.Delete, app_args=[MEMBER_DICT['resource_id'], MEMBER_DICT['resource_type'], MEMBER_DICT['member_id']])
        self.client.members.delete.assert_called_once_with(MEMBER_DICT['resource_id'], MEMBER_DICT['resource_type'], MEMBER_DICT['member_id'])