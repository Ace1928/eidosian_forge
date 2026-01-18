import json
from unittest import mock
from zaqarclient import errors
from zaqarclient.queues import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import message
from zaqarclient.queues.v2 import subscription
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
class QueuesV1_1QueueFunctionalTest(QueuesV1QueueFunctionalTest):

    def test_queue_create_functional(self):
        pass

    def test_queue_exists_functional(self):
        queue = self.client.queue('404')
        self.assertRaises(errors.InvalidOperation, queue.exists)

    def test_queue_delete_functional(self):
        queue = self.client.queue('nonono')
        queue._get_transport = mock.Mock(return_value=self.transport)
        messages = [{'ttl': 60, 'body': 'Post It 1!'}, {'ttl': 60, 'body': 'Post It 2!'}, {'ttl': 60, 'body': 'Post It 3!'}]
        queue.post(messages)
        queue.delete()
        self.assertEqual(0, len(list(queue.messages(echo=True))))

    def test_message_pop(self):
        queue = self.client.queue('test_queue')
        self.addCleanup(queue.delete)
        queue._get_transport = mock.Mock(return_value=self.transport)
        messages = [{'ttl': 60, 'body': 'Post It 1!'}, {'ttl': 60, 'body': 'Post It 2!'}, {'ttl': 60, 'body': 'Post It 2!'}]
        queue.post(messages)
        messages = queue.pop(count=2)
        self.assertIsInstance(messages, iterator._Iterator)
        self.assertEqual(2, len(list(messages)))
        remaining = queue.messages(echo=True)
        self.assertEqual(1, len(list(remaining)))

    def test_queue_metadata_functional(self):
        pass

    def test_queue_metadata_reload_functional(self):
        pass