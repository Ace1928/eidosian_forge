from openstack.network.v2 import subnet_pool
from openstack.tests.unit import base
class TestSubnetpool(base.TestCase):

    def test_basic(self):
        sot = subnet_pool.SubnetPool()
        self.assertEqual('subnetpool', sot.resource_key)
        self.assertEqual('subnetpools', sot.resources_key)
        self.assertEqual('/subnetpools', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = subnet_pool.SubnetPool(**EXAMPLE)
        self.assertEqual(EXAMPLE['address_scope_id'], sot.address_scope_id)
        self.assertEqual(EXAMPLE['created_at'], sot.created_at)
        self.assertEqual(EXAMPLE['default_prefixlen'], sot.default_prefix_length)
        self.assertEqual(EXAMPLE['default_quota'], sot.default_quota)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['ip_version'], sot.ip_version)
        self.assertTrue(sot.is_default)
        self.assertEqual(EXAMPLE['max_prefixlen'], sot.maximum_prefix_length)
        self.assertEqual(EXAMPLE['min_prefixlen'], sot.minimum_prefix_length)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['prefixes'], sot.prefixes)
        self.assertEqual(EXAMPLE['revision_number'], sot.revision_number)
        self.assertTrue(sot.is_shared)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['updated_at'], sot.updated_at)