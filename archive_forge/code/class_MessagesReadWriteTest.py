import ddt
from manilaclient.tests.functional import base
@ddt.ddt
class MessagesReadWriteTest(base.BaseTestCase):

    def setUp(self):
        super(MessagesReadWriteTest, self).setUp()
        self.message = self.create_message()

    def test_list_messages(self):
        self.skip_if_microversion_not_supported('2.37')
        messages = self.admin_client.list_messages()
        self.assertTrue(any((m['ID'] is not None for m in messages)))
        self.assertTrue(any((m['User Message'] is not None for m in messages)))
        self.assertTrue(any((m['Resource ID'] is not None for m in messages)))
        self.assertTrue(any((m['Action ID'] is not None for m in messages)))
        self.assertTrue(any((m['Detail ID'] is not None for m in messages)))
        self.assertTrue(any((m['Resource Type'] is not None for m in messages)))

    @ddt.data('id', 'action_id', 'resource_id', 'action_id', 'detail_id', 'resource_type', 'created_at', 'action_id,detail_id,resource_id')
    def test_list_share_type_select_column(self, columns):
        self.skip_if_microversion_not_supported('2.37')
        self.admin_client.list_messages(columns=columns)

    def test_get_message(self):
        self.skip_if_microversion_not_supported('2.37')
        message = self.admin_client.get_message(self.message['ID'])
        expected_keys = ('id', 'action_id', 'resource_id', 'action_id', 'detail_id', 'resource_type', 'created_at', 'created_at')
        for key in expected_keys:
            self.assertIn(key, message)

    def test_delete_message(self):
        self.skip_if_microversion_not_supported('2.37')
        message = self.create_message(cleanup_in_class=False)
        self.admin_client.delete_message(message['ID'])
        self.admin_client.wait_for_message_deletion(message['ID'])