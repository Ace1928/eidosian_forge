from unittest import mock
import ddt
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes as fake
from manilaclient.v2 import messages
@ddt.ddt
class MessageManagerTest(utils.TestCase):

    def setUp(self):
        super(MessageManagerTest, self).setUp()
        self.manager = messages.MessageManager(fake.FakeClient())

    def test_get(self):
        fake_message = fake.Message()
        mock_get = self.mock_object(self.manager, '_get', mock.Mock(return_value=fake_message))
        result = self.manager.get(fake.Message.id)
        self.assertIs(fake_message, result)
        mock_get.assert_called_once_with(messages.RESOURCE_PATH % fake.Message.id, messages.RESOURCE_NAME)

    def test_list(self):
        fake_message = fake.Message()
        mock_list = self.mock_object(self.manager, '_list', mock.Mock(return_value=[fake_message]))
        result = self.manager.list()
        self.assertEqual([fake_message], result)
        mock_list.assert_called_once_with(messages.RESOURCES_PATH, messages.RESOURCES_NAME)

    @ddt.data(({'action_id': 1, 'resource_type': 'share'}, '?action_id=1&resource_type=share'), ({'action_id': 1}, '?action_id=1'))
    @ddt.unpack
    def test_list_with_filters(self, filters, filters_path):
        fake_message = fake.Message()
        mock_list = self.mock_object(self.manager, '_list', mock.Mock(return_value=[fake_message]))
        result = self.manager.list(search_opts=filters)
        self.assertEqual([fake_message], result)
        expected_path = messages.RESOURCES_PATH + filters_path
        mock_list.assert_called_once_with(expected_path, messages.RESOURCES_NAME)

    @ddt.data('id', 'project_id', 'request_id', 'resource_type', 'action_id', 'detail_id', 'resource_id', 'message_level', 'expires_at', 'request_id', 'created_at')
    def test_list_with_sorting(self, key):
        fake_message = fake.Message()
        mock_list = self.mock_object(self.manager, '_list', mock.Mock(return_value=[fake_message]))
        result = self.manager.list(sort_dir='asc', sort_key=key)
        self.assertEqual([fake_message], result)
        expected_path = messages.RESOURCES_PATH + '?sort_dir=asc&sort_key=' + key
        mock_list.assert_called_once_with(expected_path, messages.RESOURCES_NAME)

    @ddt.data(('name', 'invalid'), ('invalid', 'asc'))
    @ddt.unpack
    def test_list_with_invalid_sorting(self, sort_key, sort_dir):
        self.assertRaises(ValueError, self.manager.list, sort_dir=sort_dir, sort_key=sort_key)

    def test_delete(self):
        mock_delete = self.mock_object(self.manager, '_delete')
        mock_post = self.mock_object(self.manager.api.client, 'post')
        self.manager.delete(fake.Message())
        mock_delete.assert_called_once_with(messages.RESOURCE_PATH % fake.Message.id)
        self.assertFalse(mock_post.called)