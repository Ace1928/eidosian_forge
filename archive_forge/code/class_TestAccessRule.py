from openstack.identity.v3 import access_rule
from openstack.tests.unit import base
class TestAccessRule(base.TestCase):

    def test_basic(self):
        sot = access_rule.AccessRule()
        self.assertEqual('access_rule', sot.resource_key)
        self.assertEqual('access_rules', sot.resources_key)
        self.assertEqual('/users/%(user_id)s/access_rules', sot.base_path)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = access_rule.AccessRule(**EXAMPLE)
        self.assertEqual(EXAMPLE['path'], sot.path)
        self.assertEqual(EXAMPLE['method'], sot.method)
        self.assertEqual(EXAMPLE['service'], sot.service)
        self.assertEqual(EXAMPLE['links'], sot.links)