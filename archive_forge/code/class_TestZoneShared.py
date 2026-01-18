import time
import uuid
from designateclient.tests import v2
class TestZoneShared(v2.APIV2TestCase, v2.CrudMixin):

    def setUp(self):
        super().setUp()
        self.zone_id = str(uuid.uuid4())
        self.target_project_id = str(uuid.uuid4())
        self.project_id = str(uuid.uuid4())
        self.created_at = time.strftime('%c')
        self.updated_at = time.strftime('%c')

    def new_ref(self, **kwargs):
        ref = super().new_ref(**kwargs)
        ref.setdefault('zone_id', self.zone_id)
        ref.setdefault('target_project_id', self.target_project_id)
        ref.setdefault('project_id', self.project_id)
        ref.setdefault('created_at', self.created_at)
        ref.setdefault('updated_at', self.updated_at)
        return ref

    def test_share_a_zone(self):
        json_body = {'target_project_id': self.target_project_id}
        expected = self.new_ref()
        self.stub_entity('POST', parts=['zones', self.zone_id, 'shares'], entity=expected, json=json_body)
        response = self.client.zone_share.create(self.zone_id, self.target_project_id)
        self.assertRequestBodyIs(json=json_body)
        self.assertEqual(expected, response)

    def test_get_zone_share(self):
        expected = self.new_ref()
        parts = ['zones', self.zone_id, 'shares']
        self.stub_entity('GET', parts=parts, entity=expected, id=expected['id'])
        response = self.client.zone_share.get(self.zone_id, expected['id'])
        self.assertRequestBodyIs(None)
        self.assertEqual(expected, response)

    def test_list_zone_shares(self):
        items = [self.new_ref(), self.new_ref()]
        parts = ['zones', self.zone_id, 'shares']
        self.stub_entity('GET', parts=parts, entity={'shared_zones': items})
        listed = self.client.zone_share.list(self.zone_id)
        self.assertList(items, listed)
        self.assertQueryStringIs('')

    def test_delete_zone_share(self):
        ref = self.new_ref()
        parts = ['zones', self.zone_id, 'shares', ref['id']]
        self.stub_url('DELETE', parts=parts)
        response = self.client.zone_share.delete(self.zone_id, ref['id'])
        self.assertRequestBodyIs(None)
        self.assertEqual('', response)