import time
import uuid
from designateclient.tests import v2
class TestZoneExports(v2.APIV2TestCase, v2.CrudMixin):

    def new_ref(self, **kwargs):
        ref = super().new_ref(**kwargs)
        ref.setdefault('zone_id', uuid.uuid4().hex)
        ref.setdefault('created_at', time.strftime('%c'))
        ref.setdefault('updated_at', time.strftime('%c'))
        ref.setdefault('status', 'PENDING')
        ref.setdefault('version', '1')
        return ref

    def test_create_export(self):
        zone = uuid.uuid4().hex
        ref = {}
        parts = ['zones', zone, 'tasks', 'export']
        self.stub_url('POST', parts=parts, json=ref)
        self.client.zone_exports.create(zone)
        self.assertRequestBodyIs(json=ref)

    def test_get_export(self):
        ref = self.new_ref()
        parts = ['zones', 'tasks', 'exports', ref['id']]
        self.stub_url('GET', parts=parts, json=ref)
        self.stub_entity('GET', parts=parts, entity=ref, id=ref['id'])
        response = self.client.zone_exports.get_export_record(ref['id'])
        self.assertEqual(ref, response)

    def test_list_exports(self):
        items = [self.new_ref(), self.new_ref()]
        parts = ['zones', 'tasks', 'exports']
        self.stub_url('GET', parts=parts, json={'exports': items})
        listed = self.client.zone_exports.list()
        self.assertList(items, listed['exports'])
        self.assertQueryStringIs('')

    def test_delete_export(self):
        ref = self.new_ref()
        parts = ['zones', 'tasks', 'exports', ref['id']]
        self.stub_url('DELETE', parts=parts, json=ref)
        self.stub_entity('DELETE', parts=parts, id=ref['id'])
        self.client.zone_exports.delete(ref['id'])
        self.assertRequestBodyIs(None)

    def test_get_export_file(self):
        ref = self.new_ref()
        parts = ['zones', 'tasks', 'exports', ref['id'], 'export']
        self.stub_url('GET', parts=parts, json=ref)
        self.stub_entity('GET', parts=parts, entity=ref, id=ref['id'])
        response = self.client.zone_exports.get_export(ref['id'])
        self.assertEqual(ref, response)