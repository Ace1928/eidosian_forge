from unittest import mock
from openstack.message.v2 import _proxy
from openstack.message.v2 import claim
from openstack.message.v2 import message
from openstack.message.v2 import queue
from openstack.message.v2 import subscription
from openstack import proxy as proxy_base
from openstack.tests.unit import test_proxy_base
class TestMessageClaim(TestMessageProxy):

    def test_claim_create(self):
        self._verify('openstack.message.v2.claim.Claim.create', self.proxy.create_claim, method_args=['test_queue'], expected_args=[self.proxy], expected_kwargs={'base_path': None})

    def test_claim_get(self):
        self._verify('openstack.proxy.Proxy._get', self.proxy.get_claim, method_args=['test_queue', 'resource_or_id'], expected_args=[claim.Claim, 'resource_or_id'], expected_kwargs={'queue_name': 'test_queue'})
        self.verify_get_overrided(self.proxy, claim.Claim, 'openstack.message.v2.claim.Claim')

    def test_claim_update(self):
        self._verify('openstack.proxy.Proxy._update', self.proxy.update_claim, method_args=['test_queue', 'resource_or_id'], method_kwargs={'k1': 'v1'}, expected_args=[claim.Claim, 'resource_or_id'], expected_kwargs={'queue_name': 'test_queue', 'k1': 'v1'})

    def test_claim_delete(self):
        self.verify_delete(self.proxy.delete_claim, claim.Claim, ignore_missing=False, method_args=['test_queue', 'test_claim'], expected_args=['test_claim'], expected_kwargs={'queue_name': 'test_queue', 'ignore_missing': False})

    def test_claim_delete_ignore(self):
        self.verify_delete(self.proxy.delete_claim, claim.Claim, ignore_missing=True, method_args=['test_queue', 'test_claim'], expected_args=['test_claim'], expected_kwargs={'queue_name': 'test_queue', 'ignore_missing': True})