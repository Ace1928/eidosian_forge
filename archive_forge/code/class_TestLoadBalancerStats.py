from unittest import mock
import uuid
from openstack.load_balancer.v2 import load_balancer
from openstack.tests.unit import base
class TestLoadBalancerStats(base.TestCase):

    def test_basic(self):
        test_load_balancer = load_balancer.LoadBalancerStats()
        self.assertEqual('stats', test_load_balancer.resource_key)
        self.assertEqual('/lbaas/loadbalancers/%(lb_id)s/stats', test_load_balancer.base_path)
        self.assertFalse(test_load_balancer.allow_create)
        self.assertTrue(test_load_balancer.allow_fetch)
        self.assertFalse(test_load_balancer.allow_delete)
        self.assertFalse(test_load_balancer.allow_list)
        self.assertFalse(test_load_balancer.allow_commit)

    def test_make_it(self):
        test_load_balancer = load_balancer.LoadBalancerStats(**EXAMPLE_STATS)
        self.assertEqual(EXAMPLE_STATS['active_connections'], test_load_balancer.active_connections)
        self.assertEqual(EXAMPLE_STATS['bytes_in'], test_load_balancer.bytes_in)
        self.assertEqual(EXAMPLE_STATS['bytes_out'], test_load_balancer.bytes_out)
        self.assertEqual(EXAMPLE_STATS['request_errors'], test_load_balancer.request_errors)
        self.assertEqual(EXAMPLE_STATS['total_connections'], test_load_balancer.total_connections)