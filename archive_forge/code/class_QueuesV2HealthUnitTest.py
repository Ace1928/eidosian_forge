import json
from unittest import mock
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
class QueuesV2HealthUnitTest(base.QueuesTestBase):

    def test_health(self):
        expect_health = {u'catalog_reachable': True, u'redis': {u'operation_status': {}, u'storage_reachable': True}}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            health_content = json.dumps(expect_health)
            health_resp = response.Response(None, health_content)
            send_method.side_effect = iter([health_resp])
            health = self.client.health()
            self.assertEqual(expect_health, health)