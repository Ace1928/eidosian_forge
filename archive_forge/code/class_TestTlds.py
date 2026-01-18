import uuid
from designateclient.tests import v2
class TestTlds(v2.APIV2TestCase, v2.CrudMixin):
    RESOURCE = 'tlds'

    def new_ref(self, **kwargs):
        ref = super().new_ref(**kwargs)
        ref.setdefault('name', uuid.uuid4().hex)
        return ref

    def test_create(self):
        ref = self.new_ref()
        self.stub_url('POST', parts=[self.RESOURCE], json=ref)
        values = ref.copy()
        del values['id']
        self.client.tlds.create(**values)
        self.assertRequestBodyIs(json=values)

    def test_create_with_description(self):
        ref = self.new_ref(description='My TLD')
        self.stub_url('POST', parts=[self.RESOURCE], json=ref)
        values = ref.copy()
        del values['id']
        self.client.tlds.create(**values)
        self.assertRequestBodyIs(json=values)

    def test_get(self):
        ref = self.new_ref()
        self.stub_entity('GET', entity=ref, id=ref['id'])
        response = self.client.tlds.get(ref['id'])
        self.assertEqual(ref, response)

    def test_get_by_name(self):
        ref = self.new_ref(name='www')
        self.stub_entity('GET', entity=ref, id=ref['id'])
        self.stub_url('GET', parts=[self.RESOURCE], json={'tlds': [ref]})
        response = self.client.tlds.get(ref['name'])
        self.assertEqual('GET', self.requests.request_history[0].method)
        self.assertEqual('http://127.0.0.1:9001/v2/tlds?name=www', self.requests.request_history[0].url)
        self.assertEqual(ref, response)

    def test_list(self):
        items = [self.new_ref(), self.new_ref()]
        self.stub_url('GET', parts=[self.RESOURCE], json={'tlds': items})
        listed = self.client.tlds.list()
        self.assertList(items, listed)
        self.assertQueryStringIs('')

    def test_update(self):
        ref = self.new_ref()
        self.stub_entity('PATCH', entity=ref, id=ref['id'])
        values = ref.copy()
        del values['id']
        self.client.tlds.update(ref['id'], values)
        self.assertRequestBodyIs(json=values)

    def test_delete(self):
        ref = self.new_ref()
        self.stub_entity('DELETE', id=ref['id'])
        self.client.tlds.delete(ref['id'])
        self.assertRequestBodyIs(None)