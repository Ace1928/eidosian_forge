import uuid
from openstack.network.v2 import qos_policy
from openstack.tests.unit import base
class TestQoSPolicy(base.TestCase):

    def test_basic(self):
        sot = qos_policy.QoSPolicy()
        self.assertEqual('policy', sot.resource_key)
        self.assertEqual('policies', sot.resources_key)
        self.assertEqual('/qos/policies', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = qos_policy.QoSPolicy(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['rules'], sot.rules)
        self.assertEqual(EXAMPLE['is_default'], sot.is_default)
        self.assertEqual(EXAMPLE['tags'], sot.tags)