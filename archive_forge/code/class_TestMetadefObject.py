from openstack.image.v2 import metadef_object
from openstack.tests.unit import base
class TestMetadefObject(base.TestCase):

    def test_basic(self):
        sot = metadef_object.MetadefObject()
        self.assertIsNone(sot.resource_key)
        self.assertEqual('objects', sot.resources_key)
        test_base_path = '/metadefs/namespaces/%(namespace_name)s/objects'
        self.assertEqual(test_base_path, sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = metadef_object.MetadefObject(**EXAMPLE)
        self.assertEqual(EXAMPLE['created_at'], sot.created_at)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['properties'], sot.properties)
        self.assertEqual(EXAMPLE['required'], sot.required)
        self.assertEqual(EXAMPLE['updated_at'], sot.updated_at)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker', 'visibility': 'visibility', 'resource_types': 'resource_types', 'sort_key': 'sort_key', 'sort_dir': 'sort_dir'}, sot._query_mapping._mapping)