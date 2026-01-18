from unittest import mock
from openstack.message.v2 import _proxy
from openstack.message.v2 import claim
from openstack.message.v2 import message
from openstack.message.v2 import queue
from openstack.message.v2 import subscription
from openstack import proxy as proxy_base
from openstack.tests.unit import test_proxy_base
class TestMessageMessage(TestMessageProxy):

    @mock.patch.object(proxy_base.Proxy, '_get_resource')
    def test_message_post(self, mock_get_resource):
        message_obj = message.Message(queue_name='test_queue')
        mock_get_resource.return_value = message_obj
        self._verify('openstack.message.v2.message.Message.post', self.proxy.post_message, method_args=['test_queue', ['msg1', 'msg2']], expected_args=[self.proxy, ['msg1', 'msg2']])
        mock_get_resource.assert_called_once_with(message.Message, None, queue_name='test_queue')

    @mock.patch.object(proxy_base.Proxy, '_get_resource')
    def test_message_get(self, mock_get_resource):
        mock_get_resource.return_value = 'resource_or_id'
        self._verify('openstack.proxy.Proxy._get', self.proxy.get_message, method_args=['test_queue', 'resource_or_id'], expected_args=[message.Message, 'resource_or_id'])
        mock_get_resource.assert_called_once_with(message.Message, 'resource_or_id', queue_name='test_queue')
        self.verify_get_overrided(self.proxy, message.Message, 'openstack.message.v2.message.Message')

    def test_messages(self):
        self.verify_list(self.proxy.messages, message.Message, method_kwargs={'queue_name': 'test_queue'}, expected_kwargs={'queue_name': 'test_queue'})

    @mock.patch.object(proxy_base.Proxy, '_get_resource')
    def test_message_delete(self, mock_get_resource):
        fake_message = mock.Mock()
        fake_message.id = 'message_id'
        mock_get_resource.return_value = fake_message
        self._verify('openstack.proxy.Proxy._delete', self.proxy.delete_message, method_args=['test_queue', 'resource_or_id', None, False], expected_args=[message.Message, fake_message], expected_kwargs={'ignore_missing': False})
        self.assertIsNone(fake_message.claim_id)
        mock_get_resource.assert_called_once_with(message.Message, 'resource_or_id', queue_name='test_queue')

    @mock.patch.object(proxy_base.Proxy, '_get_resource')
    def test_message_delete_claimed(self, mock_get_resource):
        fake_message = mock.Mock()
        fake_message.id = 'message_id'
        mock_get_resource.return_value = fake_message
        self._verify('openstack.proxy.Proxy._delete', self.proxy.delete_message, method_args=['test_queue', 'resource_or_id', 'claim_id', False], expected_args=[message.Message, fake_message], expected_kwargs={'ignore_missing': False})
        self.assertEqual('claim_id', fake_message.claim_id)
        mock_get_resource.assert_called_once_with(message.Message, 'resource_or_id', queue_name='test_queue')

    @mock.patch.object(proxy_base.Proxy, '_get_resource')
    def test_message_delete_ignore(self, mock_get_resource):
        fake_message = mock.Mock()
        fake_message.id = 'message_id'
        mock_get_resource.return_value = fake_message
        self._verify('openstack.proxy.Proxy._delete', self.proxy.delete_message, method_args=['test_queue', 'resource_or_id', None, True], expected_args=[message.Message, fake_message], expected_kwargs={'ignore_missing': True})
        self.assertIsNone(fake_message.claim_id)
        mock_get_resource.assert_called_once_with(message.Message, 'resource_or_id', queue_name='test_queue')