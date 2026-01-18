from openstack.image.v2 import metadef_resource_type
from openstack.tests.unit import base
class TestMetadefResourceType(base.TestCase):

    def test_basic(self):
        sot = metadef_resource_type.MetadefResourceType()
        self.assertIsNone(sot.resource_key)
        self.assertEqual('resource_types', sot.resources_key)
        self.assertEqual('/metadefs/resource_types', sot.base_path)
        self.assertFalse(sot.allow_create)
        self.assertFalse(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertFalse(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = metadef_resource_type.MetadefResourceType(**EXAMPLE)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['created_at'], sot.created_at)