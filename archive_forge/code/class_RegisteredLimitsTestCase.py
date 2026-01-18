import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
class RegisteredLimitsTestCase(test_v3.RestfulTestCase):
    """Test registered_limits CRUD."""

    def setUp(self):
        super(RegisteredLimitsTestCase, self).setUp()
        PROVIDERS.assignment_api.create_system_grant_for_user(self.user_id, self.role_id)
        self.system_admin_token = self.get_system_scoped_token()
        response = self.post('/regions', body={'region': {}})
        self.region2 = response.json_body['region']
        self.region_id2 = self.region2['id']
        service_ref = {'service': {'name': uuid.uuid4().hex, 'enabled': True, 'type': 'type2'}}
        response = self.post('/services', body=service_ref)
        self.service2 = response.json_body['service']
        self.service_id2 = self.service2['id']

    def test_create_registered_limit(self):
        ref = unit.new_registered_limit_ref(service_id=self.service_id, region_id=self.region_id)
        r = self.post('/registered_limits', body={'registered_limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        registered_limits = r.result['registered_limits']
        for key in ['service_id', 'region_id', 'resource_name', 'default_limit', 'description']:
            self.assertEqual(registered_limits[0][key], ref[key])

    def test_create_registered_limit_without_region(self):
        ref = unit.new_registered_limit_ref(service_id=self.service_id)
        r = self.post('/registered_limits', body={'registered_limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        registered_limits = r.result['registered_limits']
        for key in ['service_id', 'resource_name', 'default_limit']:
            self.assertEqual(registered_limits[0][key], ref[key])
        self.assertIsNone(registered_limits[0].get('region_id'))

    def test_create_registered_without_description(self):
        ref = unit.new_registered_limit_ref(service_id=self.service_id, region_id=self.region_id)
        ref.pop('description')
        r = self.post('/registered_limits', body={'registered_limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        registered_limits = r.result['registered_limits']
        for key in ['service_id', 'region_id', 'resource_name', 'default_limit']:
            self.assertEqual(registered_limits[0][key], ref[key])
        self.assertIsNone(registered_limits[0]['description'])

    def test_create_multi_registered_limit(self):
        ref1 = unit.new_registered_limit_ref(service_id=self.service_id, region_id=self.region_id, resource_name='volume')
        ref2 = unit.new_registered_limit_ref(service_id=self.service_id, resource_name='snapshot')
        r = self.post('/registered_limits', body={'registered_limits': [ref1, ref2]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        registered_limits = r.result['registered_limits']
        for key in ['service_id', 'resource_name', 'default_limit']:
            self.assertEqual(registered_limits[0][key], ref1[key])
            self.assertEqual(registered_limits[1][key], ref2[key])
        self.assertEqual(registered_limits[0]['region_id'], ref1['region_id'])
        self.assertIsNone(registered_limits[1].get('region_id'))

    def test_create_registered_limit_return_count(self):
        ref1 = unit.new_registered_limit_ref(service_id=self.service_id, region_id=self.region_id)
        r = self.post('/registered_limits', body={'registered_limits': [ref1]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        registered_limits = r.result['registered_limits']
        self.assertEqual(1, len(registered_limits))
        ref2 = unit.new_registered_limit_ref(service_id=self.service_id2, region_id=self.region_id2)
        ref3 = unit.new_registered_limit_ref(service_id=self.service_id2)
        r = self.post('/registered_limits', body={'registered_limits': [ref2, ref3]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        registered_limits = r.result['registered_limits']
        self.assertEqual(2, len(registered_limits))

    def test_create_registered_limit_with_invalid_input(self):
        ref1 = unit.new_registered_limit_ref()
        ref2 = unit.new_registered_limit_ref(default_limit='not_int')
        ref3 = unit.new_registered_limit_ref(resource_name=123)
        ref4 = unit.new_registered_limit_ref(region_id='fake_region')
        for input_limit in [ref1, ref2, ref3, ref4]:
            self.post('/registered_limits', body={'registered_limits': [input_limit]}, token=self.system_admin_token, expected_status=http.client.BAD_REQUEST)

    def test_create_registered_limit_duplicate(self):
        ref = unit.new_registered_limit_ref(service_id=self.service_id, region_id=self.region_id)
        self.post('/registered_limits', body={'registered_limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        self.post('/registered_limits', body={'registered_limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CONFLICT)

    def test_update_registered_limit(self):
        ref = unit.new_registered_limit_ref(service_id=self.service_id, region_id=self.region_id, resource_name='volume', default_limit=10)
        r = self.post('/registered_limits', body={'registered_limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        update_ref = {'service_id': self.service_id2, 'region_id': self.region_id2, 'resource_name': 'snapshot', 'default_limit': 5, 'description': 'test description'}
        r = self.patch('/registered_limits/%s' % r.result['registered_limits'][0]['id'], body={'registered_limit': update_ref}, token=self.system_admin_token, expected_status=http.client.OK)
        new_registered_limits = r.result['registered_limit']
        self.assertEqual(new_registered_limits['service_id'], self.service_id2)
        self.assertEqual(new_registered_limits['region_id'], self.region_id2)
        self.assertEqual(new_registered_limits['resource_name'], 'snapshot')
        self.assertEqual(new_registered_limits['default_limit'], 5)
        self.assertEqual(new_registered_limits['description'], 'test description')

    def test_update_registered_limit_region_failed(self):
        ref = unit.new_registered_limit_ref(service_id=self.service_id, resource_name='volume', default_limit=10, description='test description')
        r = self.post('/registered_limits', body={'registered_limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        update_ref = {'region_id': self.region_id}
        registered_limit_id = r.result['registered_limits'][0]['id']
        r = self.patch('/registered_limits/%s' % registered_limit_id, body={'registered_limit': update_ref}, token=self.system_admin_token, expected_status=http.client.OK)
        new_registered_limits = r.result['registered_limit']
        self.assertEqual(self.region_id, new_registered_limits['region_id'])
        update_ref['region_id'] = ''
        r = self.patch('/registered_limits/%s' % registered_limit_id, body={'registered_limit': update_ref}, token=self.system_admin_token, expected_status=http.client.BAD_REQUEST)

    def test_update_registered_limit_description(self):
        ref = unit.new_registered_limit_ref(service_id=self.service_id, region_id=self.region_id, resource_name='volume', default_limit=10)
        r = self.post('/registered_limits', body={'registered_limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        update_ref = {'description': 'test description'}
        registered_limit_id = r.result['registered_limits'][0]['id']
        r = self.patch('/registered_limits/%s' % registered_limit_id, body={'registered_limit': update_ref}, token=self.system_admin_token, expected_status=http.client.OK)
        new_registered_limits = r.result['registered_limit']
        self.assertEqual(new_registered_limits['description'], 'test description')
        update_ref['description'] = ''
        r = self.patch('/registered_limits/%s' % registered_limit_id, body={'registered_limit': update_ref}, token=self.system_admin_token, expected_status=http.client.OK)
        new_registered_limits = r.result['registered_limit']
        self.assertEqual(new_registered_limits['description'], '')

    def test_update_registered_limit_region_id_to_none(self):
        ref = unit.new_registered_limit_ref(service_id=self.service_id, region_id=self.region_id, resource_name='volume', default_limit=10)
        r = self.post('/registered_limits', body={'registered_limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        update_ref = {'region_id': None}
        registered_limit_id = r.result['registered_limits'][0]['id']
        r = self.patch('/registered_limits/%s' % registered_limit_id, body={'registered_limit': update_ref}, token=self.system_admin_token, expected_status=http.client.OK)
        self.assertIsNone(r.result['registered_limit']['region_id'])

    def test_update_registered_limit_region_id_to_none_conflict(self):
        ref1 = unit.new_registered_limit_ref(service_id=self.service_id, resource_name='volume', default_limit=10)
        ref2 = unit.new_registered_limit_ref(service_id=self.service_id, region_id=self.region_id, resource_name='volume', default_limit=10)
        self.post('/registered_limits', body={'registered_limits': [ref1]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        r = self.post('/registered_limits', body={'registered_limits': [ref2]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        update_ref = {'region_id': None}
        registered_limit_id = r.result['registered_limits'][0]['id']
        self.patch('/registered_limits/%s' % registered_limit_id, body={'registered_limit': update_ref}, token=self.system_admin_token, expected_status=http.client.CONFLICT)

    def test_update_registered_limit_not_found(self):
        update_ref = {'service_id': self.service_id, 'region_id': self.region_id, 'resource_name': 'snapshot', 'default_limit': 5}
        self.patch('/registered_limits/%s' % uuid.uuid4().hex, body={'registered_limit': update_ref}, token=self.system_admin_token, expected_status=http.client.NOT_FOUND)

    def test_update_registered_limit_with_invalid_input(self):
        ref = unit.new_registered_limit_ref(service_id=self.service_id, region_id=self.region_id, resource_name='volume', default_limit=10)
        r = self.post('/registered_limits', body={'registered_limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        reg_id = r.result['registered_limits'][0]['id']
        update_ref1 = unit.new_registered_limit_ref(service_id='fake_id')
        update_ref2 = unit.new_registered_limit_ref(default_limit='not_int')
        update_ref3 = unit.new_registered_limit_ref(resource_name=123)
        update_ref4 = unit.new_registered_limit_ref(region_id='fake_region')
        update_ref5 = unit.new_registered_limit_ref(description=123)
        for input_limit in [update_ref1, update_ref2, update_ref3, update_ref4, update_ref5]:
            self.patch('/registered_limits/%s' % reg_id, body={'registered_limit': input_limit}, token=self.system_admin_token, expected_status=http.client.BAD_REQUEST)

    def test_update_registered_limit_with_referenced_limit(self):
        ref = unit.new_registered_limit_ref(service_id=self.service_id, region_id=self.region_id, resource_name='volume', default_limit=10)
        r = self.post('/registered_limits', body={'registered_limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        ref = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume')
        self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        update_ref = {'service_id': self.service_id2, 'region_id': self.region_id2, 'resource_name': 'snapshot', 'default_limit': 5}
        self.patch('/registered_limits/%s' % r.result['registered_limits'][0]['id'], body={'registered_limit': update_ref}, token=self.system_admin_token, expected_status=http.client.FORBIDDEN)

    def test_list_registered_limit(self):
        r = self.get('/registered_limits', expected_status=http.client.OK)
        self.assertEqual([], r.result.get('registered_limits'))
        ref1 = unit.new_registered_limit_ref(service_id=self.service_id, resource_name='test_resource', region_id=self.region_id)
        ref2 = unit.new_registered_limit_ref(service_id=self.service_id2, resource_name='test_resource', region_id=self.region_id2)
        r = self.post('/registered_limits', body={'registered_limits': [ref1, ref2]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        id1 = r.result['registered_limits'][0]['id']
        r = self.get('/registered_limits', expected_status=http.client.OK)
        registered_limits = r.result['registered_limits']
        self.assertEqual(len(registered_limits), 2)
        for key in ['service_id', 'region_id', 'resource_name', 'default_limit']:
            if registered_limits[0]['id'] == id1:
                self.assertEqual(registered_limits[0][key], ref1[key])
                self.assertEqual(registered_limits[1][key], ref2[key])
                break
            self.assertEqual(registered_limits[1][key], ref1[key])
            self.assertEqual(registered_limits[0][key], ref2[key])
        r = self.get('/registered_limits?service_id=%s' % self.service_id, expected_status=http.client.OK)
        registered_limits = r.result['registered_limits']
        self.assertEqual(len(registered_limits), 1)
        for key in ['service_id', 'region_id', 'resource_name', 'default_limit']:
            self.assertEqual(registered_limits[0][key], ref1[key])
        r = self.get('/registered_limits?region_id=%s' % self.region_id2, expected_status=http.client.OK)
        registered_limits = r.result['registered_limits']
        self.assertEqual(len(registered_limits), 1)
        for key in ['service_id', 'region_id', 'resource_name', 'default_limit']:
            self.assertEqual(registered_limits[0][key], ref2[key])
        r = self.get('/registered_limits?resource_name=test_resource', expected_status=http.client.OK)
        registered_limits = r.result['registered_limits']
        self.assertEqual(len(registered_limits), 2)

    def test_show_registered_limit(self):
        ref1 = unit.new_registered_limit_ref(service_id=self.service_id, region_id=self.region_id)
        ref2 = unit.new_registered_limit_ref(service_id=self.service_id2, region_id=self.region_id2)
        r = self.post('/registered_limits', body={'registered_limits': [ref1, ref2]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        id1 = r.result['registered_limits'][0]['id']
        self.get('/registered_limits/fake_id', expected_status=http.client.NOT_FOUND)
        r = self.get('/registered_limits/%s' % id1, expected_status=http.client.OK)
        registered_limit = r.result['registered_limit']
        for key in ['service_id', 'region_id', 'resource_name', 'default_limit', 'description']:
            self.assertEqual(registered_limit[key], ref1[key])

    def test_delete_registered_limit(self):
        ref1 = unit.new_registered_limit_ref(service_id=self.service_id, region_id=self.region_id)
        ref2 = unit.new_registered_limit_ref(service_id=self.service_id2, region_id=self.region_id2)
        r = self.post('/registered_limits', body={'registered_limits': [ref1, ref2]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        id1 = r.result['registered_limits'][0]['id']
        self.delete('/registered_limits/%s' % id1, token=self.system_admin_token, expected_status=http.client.NO_CONTENT)
        self.delete('/registered_limits/fake_id', token=self.system_admin_token, expected_status=http.client.NOT_FOUND)
        r = self.get('/registered_limits', expected_status=http.client.OK)
        registered_limits = r.result['registered_limits']
        self.assertEqual(len(registered_limits), 1)

    def test_delete_registered_limit_with_referenced_limit(self):
        ref = unit.new_registered_limit_ref(service_id=self.service_id, region_id=self.region_id, resource_name='volume', default_limit=10)
        r = self.post('/registered_limits', body={'registered_limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        ref = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume')
        self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        id = r.result['registered_limits'][0]['id']
        self.delete('/registered_limits/%s' % id, expected_status=http.client.FORBIDDEN)