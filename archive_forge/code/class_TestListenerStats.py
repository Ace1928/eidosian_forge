import uuid
from openstack.load_balancer.v2 import listener
from openstack.tests.unit import base
class TestListenerStats(base.TestCase):

    def test_basic(self):
        test_listener = listener.ListenerStats()
        self.assertEqual('stats', test_listener.resource_key)
        self.assertEqual('/lbaas/listeners/%(listener_id)s/stats', test_listener.base_path)
        self.assertFalse(test_listener.allow_create)
        self.assertTrue(test_listener.allow_fetch)
        self.assertFalse(test_listener.allow_delete)
        self.assertFalse(test_listener.allow_list)
        self.assertFalse(test_listener.allow_commit)

    def test_make_it(self):
        test_listener = listener.ListenerStats(**EXAMPLE_STATS)
        self.assertEqual(EXAMPLE_STATS['active_connections'], test_listener.active_connections)
        self.assertEqual(EXAMPLE_STATS['bytes_in'], test_listener.bytes_in)
        self.assertEqual(EXAMPLE_STATS['bytes_out'], test_listener.bytes_out)
        self.assertEqual(EXAMPLE_STATS['request_errors'], test_listener.request_errors)
        self.assertEqual(EXAMPLE_STATS['total_connections'], test_listener.total_connections)