from openstack.shared_file_system.v2 import share_export_locations as el
from openstack.tests.unit import base
class TestShareExportLocations(base.TestCase):

    def test_basic(self):
        export = el.ShareExportLocation()
        self.assertEqual('export_locations', export.resources_key)
        self.assertEqual('/shares/%(share_id)s/export_locations', export.base_path)
        self.assertTrue(export.allow_list)

    def test_share_export_locations(self):
        export = el.ShareExportLocation(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], export.id)
        self.assertEqual(EXAMPLE['path'], export.path)
        self.assertEqual(EXAMPLE['preferred'], export.is_preferred)
        self.assertEqual(EXAMPLE['share_instance_id'], export.share_instance_id)
        self.assertEqual(EXAMPLE['is_admin_only'], export.is_admin)