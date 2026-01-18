from unittest import mock
from keystoneauth1 import adapter
from openstack.dns.v2 import zone_import
from openstack.tests.unit import base
@mock.patch.object(zone_import.ZoneImport, '_translate_response', mock.Mock())
class TestZoneImport(base.TestCase):

    def test_basic(self):
        sot = zone_import.ZoneImport()
        self.assertEqual('', sot.resource_key)
        self.assertEqual('imports', sot.resources_key)
        self.assertEqual('/zones/tasks/import', sot.base_path)
        self.assertTrue(sot.allow_list)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker', 'message': 'message', 'status': 'status', 'zone_id': 'zone_id'}, sot._query_mapping._mapping)

    def test_make_it(self):
        sot = zone_import.ZoneImport(**EXAMPLE)
        self.assertEqual(IDENTIFIER, sot.id)
        self.assertEqual(EXAMPLE['created_at'], sot.created_at)
        self.assertEqual(EXAMPLE['updated_at'], sot.updated_at)
        self.assertEqual(EXAMPLE['version'], sot.version)
        self.assertEqual(EXAMPLE['message'], sot.message)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['status'], sot.status)
        self.assertEqual(EXAMPLE['zone_id'], sot.zone_id)

    def test_create(self):
        sot = zone_import.ZoneImport()
        response = mock.Mock()
        response.json = mock.Mock(return_value='')
        self.session = mock.Mock(spec=adapter.Adapter)
        self.session.default_microversion = '1.1'
        sot.create(self.session)
        self.session.post.assert_called_once_with(mock.ANY, json=None, headers={'content-type': 'text/dns'}, microversion=self.session.default_microversion)