from openstack.network.v2 import extension
from openstack.tests.unit import base
class TestExtension(base.TestCase):

    def test_basic(self):
        sot = extension.Extension()
        self.assertEqual('extension', sot.resource_key)
        self.assertEqual('extensions', sot.resources_key)
        self.assertEqual('/extensions', sot.base_path)
        self.assertFalse(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertFalse(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = extension.Extension(**EXAMPLE)
        self.assertEqual(EXAMPLE['alias'], sot.id)
        self.assertEqual(EXAMPLE['alias'], sot.alias)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['links'], sot.links)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['updated'], sot.updated_at)