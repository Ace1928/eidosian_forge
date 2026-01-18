import json
import time
from unittest import mock
from zaqarclient.queues.v1 import claim
from zaqarclient.tests.queues import base
from zaqarclient.transport import errors
from zaqarclient.transport import response
class QueueV1_1ClaimUnitTest(QueueV1ClaimUnitTest):

    def test_claim(self):
        result = [{'href': '/v1/queues/fizbit/messages/50b68a50d6f5b8c8a7c62b01', 'ttl': 800, 'age': 790, 'body': {'event': 'ActivateAccount', 'mode': 'active'}}, {'href': '/v1/queues/fizbit/messages/50b68a50d6f5b8c8a7c62b02', 'ttl': 800, 'age': 790, 'body': {'event': 'ActivateAccount', 'mode': 'active'}}]
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, json.dumps({'messages': result}))
            send_method.return_value = resp
            claimed = self.queue.claim(ttl=60, grace=60)
            num_tested = 0
            for num, msg in enumerate(claimed):
                num_tested += 1
                self.assertEqual(result[num]['href'], msg.href)
            self.assertEqual(len(result), num_tested)