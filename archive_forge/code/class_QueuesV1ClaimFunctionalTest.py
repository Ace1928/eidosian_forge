import json
import time
from unittest import mock
from zaqarclient.queues.v1 import claim
from zaqarclient.tests.queues import base
from zaqarclient.transport import errors
from zaqarclient.transport import response
class QueuesV1ClaimFunctionalTest(base.QueuesTestBase):

    def test_message_claim_functional(self):
        queue = self.client.queue('test_queue')
        queue._get_transport = mock.Mock(return_value=self.transport)
        messages = [{'ttl': 60, 'body': 'Post It 1!'}]
        queue.post(messages)
        messages = queue.claim(ttl=120, grace=120)
        self.assertIsInstance(messages, claim.Claim)
        self.assertGreaterEqual(len(list(messages)), 0)

    def test_claim_get_functional(self):
        queue = self.client.queue('test_queue')
        queue._get_transport = mock.Mock(return_value=self.transport)
        res = queue.claim(ttl=100, grace=100)
        claim_id = res.id
        cl = queue.claim(id=claim_id)
        self.assertEqual(claim_id, cl.id)

    def test_claim_create_delete_functional(self):
        queue = self.client.queue('test_queue')
        queue._get_transport = mock.Mock(return_value=self.transport)
        messages = [{'ttl': 60, 'body': 'Post It 1!'}]
        queue.post(messages)
        cl = queue.claim(ttl=120, grace=120)
        claim_id = cl.id
        cl.delete()
        self.assertRaises(errors.ResourceNotFound, queue.claim, id=claim_id)

    def test_claim_age(self):
        queue = self.client.queue('test_queue')
        queue._get_transport = mock.Mock(return_value=self.transport)
        messages = [{'ttl': 60, 'body': 'image.upload'}]
        queue.post(messages)
        res = queue.claim(ttl=100, grace=60)
        self.assertGreaterEqual(res.age, 0)
        time.sleep(2)
        self.assertGreaterEqual(res.age, 2)