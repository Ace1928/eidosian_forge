import uuid
from openstack.load_balancer.v2 import amphora
from openstack.tests.unit import base
class TestAmphoraFailover(base.TestCase):

    def test_basic(self):
        test_amp_failover = amphora.AmphoraFailover()
        self.assertEqual('/octavia/amphorae/%(amphora_id)s/failover', test_amp_failover.base_path)
        self.assertFalse(test_amp_failover.allow_create)
        self.assertFalse(test_amp_failover.allow_fetch)
        self.assertTrue(test_amp_failover.allow_commit)
        self.assertFalse(test_amp_failover.allow_delete)
        self.assertFalse(test_amp_failover.allow_list)