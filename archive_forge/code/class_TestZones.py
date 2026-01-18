import time
import uuid
from designateclient.tests import v2
class TestZones(v2.APIV2TestCase, v2.CrudMixin):
    RESOURCE = 'zones'

    def new_ref(self, **kwargs):
        ref = super().new_ref(**kwargs)
        ref.setdefault('name', uuid.uuid4().hex)
        ref.setdefault('type', 'PRIMARY')
        return ref

    def test_create_with_description(self):
        ref = self.new_ref(email='root@example.com', description='Foo')
        self.stub_url('POST', parts=[self.RESOURCE], json=ref)
        values = ref.copy()
        del values['id']
        self.client.zones.create(values['name'], email=values['email'], description=values['description'])
        self.assertRequestBodyIs(json=values)

    def test_create_primary(self):
        ref = self.new_ref(email='root@example.com')
        self.stub_url('POST', parts=[self.RESOURCE], json=ref)
        values = ref.copy()
        del values['id']
        self.client.zones.create(values['name'], email=values['email'])
        self.assertRequestBodyIs(json=values)

    def test_create_primary_with_ttl(self):
        ref = self.new_ref(email='root@example.com', ttl=60)
        self.stub_url('POST', parts=[self.RESOURCE], json=ref)
        values = ref.copy()
        del values['id']
        self.client.zones.create(values['name'], email=values['email'], ttl=values['ttl'])
        self.assertRequestBodyIs(json=values)

    def test_create_secondary(self):
        ref = self.new_ref(type='SECONDARY', masters=['10.0.0.1'])
        self.stub_url('POST', parts=[self.RESOURCE], json=ref)
        values = ref.copy()
        del values['id']
        self.client.zones.create(values['name'], type_=values['type'], masters=values['masters'])
        self.assertRequestBodyIs(json=values)

    def test_get(self):
        ref = self.new_ref()
        self.stub_entity('GET', entity=ref, id=ref['id'])
        response = self.client.zones.get(ref['id'])
        self.assertEqual(ref, response)

    def test_list(self):
        items = [self.new_ref(), self.new_ref()]
        self.stub_url('GET', parts=[self.RESOURCE], json={'zones': items})
        listed = self.client.zones.list()
        self.assertList(items, listed)
        self.assertQueryStringIs('')

    def test_update(self):
        ref = self.new_ref()
        self.stub_entity('PATCH', entity=ref, id=ref['id'])
        values = ref.copy()
        del values['id']
        self.client.zones.update(ref['id'], values)
        self.assertRequestBodyIs(json=values)

    def test_delete(self):
        ref = self.new_ref()
        self.stub_entity('DELETE', id=ref['id'])
        self.client.zones.delete(ref['id'])
        self.assertRequestBodyIs(None)
        self.assertRequestHeaderEqual('X-Designate-Delete-Shares', None)

    def test_delete_with_delete_shares(self):
        ref = self.new_ref()
        self.stub_entity('DELETE', id=ref['id'])
        self.client.zones.delete(ref['id'], delete_shares=True)
        self.assertRequestBodyIs(None)
        self.assertRequestHeaderEqual('X-Designate-Delete-Shares', 'true')

    def test_task_abandon(self):
        ref = self.new_ref()
        parts = [self.RESOURCE, ref['id'], 'tasks', 'abandon']
        self.stub_url('POST', parts=parts)
        self.client.zones.abandon(ref['id'])
        self.assertRequestBodyIs(None)

    def test_task_axfr(self):
        ref = self.new_ref()
        parts = [self.RESOURCE, ref['id'], 'tasks', 'xfr']
        self.stub_url('POST', parts=parts)
        self.client.zones.axfr(ref['id'])
        self.assertRequestBodyIs(None)

    def test_task_pool_move(self):
        ref = self.new_ref(pool_id=1)
        parts = [self.RESOURCE, ref['id'], 'tasks', 'pool_move']
        self.stub_url('POST', parts=parts)
        values = ref.copy()
        self.client.zones.pool_move(ref['id'], values)
        self.assertRequestBodyIs(json=values)