import json
import time
from unittest import mock
from zaqarclient.queues.v1 import claim
from zaqarclient.tests.queues import base
from zaqarclient.transport import errors
from zaqarclient.transport import response
class QueueV1ClaimUnitTest(base.QueuesTestBase):

    def test_claim(self):
        result = [{'href': '/v1/queues/fizbit/messages/50b68a50d6f5b8c8a7c62b01', 'ttl': 800, 'age': 790, 'body': {'event': 'ActivateAccount', 'mode': 'active'}}, {'href': '/v1/queues/fizbit/messages/50b68a50d6f5b8c8a7c62b02', 'ttl': 800, 'age': 790, 'body': {'event': 'ActivateAccount', 'mode': 'active'}}]
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, json.dumps(result))
            send_method.return_value = resp
            claimed = self.queue.claim(ttl=60, grace=60)
            num_tested = 0
            for num, msg in enumerate(claimed):
                num_tested += 1
                self.assertEqual(result[num]['href'], msg.href)
            self.assertEqual(len(result), num_tested)

    def test_claim_limit(self):

        def verify_limit(request):
            self.assertIn('limit', request.params)
            self.assertEqual(10, request.params['limit'])
            return response.Response(None, "{0: [], 'messages': []}")
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            send_method.side_effect = verify_limit
            self.queue.claim(ttl=60, grace=60, limit=10)

    def test_claim_get_by_id(self):
        result = {'href': '/v1/queues/fizbit/messages/50b68a50d6cb01?claim_id=4524', 'age': 790, 'ttl': 800, 'messages': [{'href': '/v1/queues/fizbit/messages/50b68a50d6f5b8c8a7c62b01', 'ttl': 800, 'age': 790, 'body': {'event': 'ActivateAccount', 'mode': 'active'}}]}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, json.dumps(result))
            send_method.return_value = resp
            cl = self.queue.claim(id='5245432')
            num_tested = 0
            for num, msg in enumerate(cl):
                num_tested += 1
                self.assertEqual(result['messages'][num]['href'], msg.href)
            self.assertEqual(len(result['messages']), num_tested)

    def test_claim_update(self):
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, None)
            send_method.return_value = resp
            self.queue.claim(id='5245432').update(ttl=444, grace=987)

    def test_claim_delete(self):
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, None)
            send_method.return_value = resp
            self.queue.claim(id='4225').delete()