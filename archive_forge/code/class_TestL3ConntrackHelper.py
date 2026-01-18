from openstack.network.v2 import l3_conntrack_helper
from openstack.tests.unit import base
class TestL3ConntrackHelper(base.TestCase):

    def test_basic(self):
        sot = l3_conntrack_helper.ConntrackHelper()
        self.assertEqual('conntrack_helper', sot.resource_key)
        self.assertEqual('conntrack_helpers', sot.resources_key)
        self.assertEqual('/routers/%(router_id)s/conntrack_helpers', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = l3_conntrack_helper.ConntrackHelper(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['protocol'], sot.protocol)
        self.assertEqual(EXAMPLE['port'], sot.port)
        self.assertEqual(EXAMPLE['helper'], sot.helper)