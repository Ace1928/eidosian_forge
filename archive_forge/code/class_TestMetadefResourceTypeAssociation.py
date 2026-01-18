from openstack.image.v2 import metadef_resource_type
from openstack.tests.unit import base
class TestMetadefResourceTypeAssociation(base.TestCase):

    def test_basic(self):
        sot = metadef_resource_type.MetadefResourceTypeAssociation()
        self.assertIsNone(sot.resource_key)
        self.assertEqual('resource_type_associations', sot.resources_key)
        self.assertEqual('/metadefs/namespaces/%(namespace_name)s/resource_types', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertFalse(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = metadef_resource_type.MetadefResourceTypeAssociation(**EXAMPLE)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['created_at'], sot.created_at)
        self.assertEqual(EXAMPLE['prefix'], sot.prefix)
        self.assertEqual(EXAMPLE['properties_target'], sot.properties_target)