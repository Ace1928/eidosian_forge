import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
class LimitsTestCase(test_v3.RestfulTestCase):
    """Test limits CRUD."""

    def setUp(self):
        super(LimitsTestCase, self).setUp()
        reader_role = {'id': uuid.uuid4().hex, 'name': 'reader'}
        reader_role = PROVIDERS.role_api.create_role(reader_role['id'], reader_role)
        member_role = {'id': uuid.uuid4().hex, 'name': 'member'}
        member_role = PROVIDERS.role_api.create_role(member_role['id'], member_role)
        PROVIDERS.role_api.create_implied_role(self.role_id, member_role['id'])
        PROVIDERS.role_api.create_implied_role(member_role['id'], reader_role['id'])
        PROVIDERS.assignment_api.create_system_grant_for_user(self.user_id, self.role_id)
        self.system_admin_token = self.get_system_scoped_token()
        response = self.post('/regions', body={'region': {}})
        self.region2 = response.json_body['region']
        self.region_id2 = self.region2['id']
        service_ref = {'service': {'name': uuid.uuid4().hex, 'enabled': True, 'type': 'type2'}}
        response = self.post('/services', body=service_ref)
        self.service2 = response.json_body['service']
        self.service_id2 = self.service2['id']
        ref1 = unit.new_registered_limit_ref(service_id=self.service_id, region_id=self.region_id, resource_name='volume')
        ref2 = unit.new_registered_limit_ref(service_id=self.service_id2, resource_name='snapshot')
        ref3 = unit.new_registered_limit_ref(service_id=self.service_id, region_id=self.region_id, resource_name='backup')
        self.post('/registered_limits', body={'registered_limits': [ref1, ref2, ref3]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        self.project_2 = unit.new_project_ref(domain_id=self.domain_id)
        self.project_2_id = self.project_2['id']
        PROVIDERS.resource_api.create_project(self.project_2_id, self.project_2)
        self.domain_2 = unit.new_domain_ref()
        self.domain_2_id = self.domain_2['id']
        PROVIDERS.resource_api.create_domain(self.domain_2_id, self.domain_2)
        self.role_2 = unit.new_role_ref(name='non-admin')
        self.role_2_id = self.role_2['id']
        PROVIDERS.role_api.create_role(self.role_2_id, self.role_2)
        PROVIDERS.assignment_api.create_grant(self.role_2_id, user_id=self.user_id, project_id=self.project_2_id)
        PROVIDERS.assignment_api.create_grant(self.role_id, user_id=self.user_id, domain_id=self.domain_id)
        PROVIDERS.assignment_api.create_grant(self.role_2_id, user_id=self.user_id, domain_id=self.domain_2_id)
        PROVIDERS.assignment_api.create_system_grant_for_user(self.user_id, self.role_id)

    def test_create_project_limit(self):
        ref = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume')
        r = self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        limits = r.result['limits']
        self.assertIsNotNone(limits[0]['id'])
        self.assertIsNone(limits[0]['domain_id'])
        for key in ['service_id', 'region_id', 'resource_name', 'resource_limit', 'description', 'project_id']:
            self.assertEqual(limits[0][key], ref[key])

    def test_create_domain_limit(self):
        ref = unit.new_limit_ref(domain_id=self.domain_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume')
        r = self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        limits = r.result['limits']
        self.assertIsNotNone(limits[0]['id'])
        self.assertIsNone(limits[0]['project_id'])
        for key in ['service_id', 'region_id', 'resource_name', 'resource_limit', 'description', 'domain_id']:
            self.assertEqual(limits[0][key], ref[key])

    def test_create_limit_without_region(self):
        ref = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id2, resource_name='snapshot')
        r = self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        limits = r.result['limits']
        self.assertIsNotNone(limits[0]['id'])
        self.assertIsNotNone(limits[0]['project_id'])
        for key in ['service_id', 'resource_name', 'resource_limit']:
            self.assertEqual(limits[0][key], ref[key])
        self.assertIsNone(limits[0].get('region_id'))

    def test_create_limit_without_description(self):
        ref = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume')
        ref.pop('description')
        r = self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        limits = r.result['limits']
        self.assertIsNotNone(limits[0]['id'])
        self.assertIsNotNone(limits[0]['project_id'])
        for key in ['service_id', 'region_id', 'resource_name', 'resource_limit']:
            self.assertEqual(limits[0][key], ref[key])
        self.assertIsNone(limits[0]['description'])

    def test_create_limit_with_domain_as_project(self):
        ref = unit.new_limit_ref(project_id=self.domain_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume')
        r = self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token)
        limits = r.result['limits']
        self.assertIsNone(limits[0]['project_id'])
        self.assertEqual(self.domain_id, limits[0]['domain_id'])

    def test_create_multi_limit(self):
        ref1 = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume')
        ref2 = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id2, resource_name='snapshot')
        r = self.post('/limits', body={'limits': [ref1, ref2]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        limits = r.result['limits']
        for key in ['service_id', 'resource_name', 'resource_limit']:
            self.assertEqual(limits[0][key], ref1[key])
            self.assertEqual(limits[1][key], ref2[key])
        self.assertEqual(limits[0]['region_id'], ref1['region_id'])
        self.assertIsNone(limits[1].get('region_id'))

    def test_create_limit_return_count(self):
        ref1 = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume')
        r = self.post('/limits', body={'limits': [ref1]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        limits = r.result['limits']
        self.assertEqual(1, len(limits))
        ref2 = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id2, resource_name='snapshot')
        ref3 = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id, region_id=self.region_id, resource_name='backup')
        r = self.post('/limits', body={'limits': [ref2, ref3]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        limits = r.result['limits']
        self.assertEqual(2, len(limits))

    def test_create_limit_with_invalid_input(self):
        ref1 = unit.new_limit_ref(project_id=self.project_id, resource_limit='not_int')
        ref2 = unit.new_limit_ref(project_id=self.project_id, resource_name=123)
        ref3 = unit.new_limit_ref(project_id=self.project_id, region_id='fake_region')
        for input_limit in [ref1, ref2, ref3]:
            self.post('/limits', body={'limits': [input_limit]}, token=self.system_admin_token, expected_status=http.client.BAD_REQUEST)

    def test_create_limit_duplicate(self):
        ref = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume')
        self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CONFLICT)

    def test_create_limit_without_reference_registered_limit(self):
        ref = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id, region_id=self.region_id2, resource_name='volume')
        self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.FORBIDDEN)

    def test_update_limit(self):
        ref = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=10)
        r = self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        update_ref = {'resource_limit': 5, 'description': 'test description'}
        r = self.patch('/limits/%s' % r.result['limits'][0]['id'], body={'limit': update_ref}, token=self.system_admin_token, expected_status=http.client.OK)
        new_limits = r.result['limit']
        self.assertEqual(new_limits['resource_limit'], 5)
        self.assertEqual(new_limits['description'], 'test description')

    def test_update_limit_not_found(self):
        update_ref = {'resource_limit': 5}
        self.patch('/limits/%s' % uuid.uuid4().hex, body={'limit': update_ref}, token=self.system_admin_token, expected_status=http.client.NOT_FOUND)

    def test_update_limit_with_invalid_input(self):
        ref = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=10)
        r = self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        limit_id = r.result['limits'][0]['id']
        invalid_resource_limit_update = {'resource_limit': 'not_int'}
        invalid_description_update = {'description': 123}
        for input_limit in [invalid_resource_limit_update, invalid_description_update]:
            self.patch('/limits/%s' % limit_id, body={'limit': input_limit}, token=self.system_admin_token, expected_status=http.client.BAD_REQUEST)

    def test_list_limit(self):
        r = self.get('/limits', token=self.system_admin_token, expected_status=http.client.OK)
        self.assertEqual([], r.result.get('limits'))
        ref1 = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume')
        ref2 = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id2, resource_name='snapshot')
        r = self.post('/limits', body={'limits': [ref1, ref2]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        id1 = r.result['limits'][0]['id']
        r = self.get('/limits', expected_status=http.client.OK)
        limits = r.result['limits']
        self.assertEqual(len(limits), 2)
        if limits[0]['id'] == id1:
            self.assertEqual(limits[0]['region_id'], ref1['region_id'])
            self.assertIsNone(limits[1].get('region_id'))
            for key in ['service_id', 'resource_name', 'resource_limit']:
                self.assertEqual(limits[0][key], ref1[key])
                self.assertEqual(limits[1][key], ref2[key])
        else:
            self.assertEqual(limits[1]['region_id'], ref1['region_id'])
            self.assertIsNone(limits[0].get('region_id'))
            for key in ['service_id', 'resource_name', 'resource_limit']:
                self.assertEqual(limits[1][key], ref1[key])
                self.assertEqual(limits[0][key], ref2[key])
        r = self.get('/limits?service_id=%s' % self.service_id2, expected_status=http.client.OK)
        limits = r.result['limits']
        self.assertEqual(len(limits), 1)
        for key in ['service_id', 'resource_name', 'resource_limit']:
            self.assertEqual(limits[0][key], ref2[key])
        r = self.get('/limits?region_id=%s' % self.region_id, expected_status=http.client.OK)
        limits = r.result['limits']
        self.assertEqual(len(limits), 1)
        for key in ['service_id', 'region_id', 'resource_name', 'resource_limit']:
            self.assertEqual(limits[0][key], ref1[key])
        r = self.get('/limits?resource_name=volume', expected_status=http.client.OK)
        limits = r.result['limits']
        self.assertEqual(len(limits), 1)
        for key in ['service_id', 'region_id', 'resource_name', 'resource_limit']:
            self.assertEqual(limits[0][key], ref1[key])

    def test_list_limit_with_project_id_filter(self):
        self.config_fixture.config(group='oslo_policy', enforce_scope=True)
        ref1 = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume')
        ref2 = unit.new_limit_ref(project_id=self.project_2_id, service_id=self.service_id2, resource_name='snapshot')
        self.post('/limits', body={'limits': [ref1, ref2]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        r = self.get('/limits', expected_status=http.client.OK)
        limits = r.result['limits']
        self.assertEqual(1, len(limits))
        self.assertEqual(self.project_id, limits[0]['project_id'])
        r = self.get('/limits', expected_status=http.client.OK, auth=self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project_2_id))
        limits = r.result['limits']
        self.assertEqual(1, len(limits))
        self.assertEqual(self.project_2_id, limits[0]['project_id'])
        r = self.get('/limits?project_id=%s' % self.project_id, expected_status=http.client.OK)
        limits = r.result['limits']
        self.assertEqual(1, len(limits))
        self.assertEqual(self.project_id, limits[0]['project_id'])
        r = self.get('/limits?project_id=%s' % self.project_id, expected_status=http.client.OK, token=self.system_admin_token)
        limits = r.result['limits']
        self.assertEqual(1, len(limits))
        self.assertEqual(self.project_id, limits[0]['project_id'])

    def test_list_limit_with_domain_id_filter(self):
        ref1 = unit.new_limit_ref(domain_id=self.domain_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume')
        ref2 = unit.new_limit_ref(domain_id=self.domain_2_id, service_id=self.service_id2, resource_name='snapshot')
        self.post('/limits', body={'limits': [ref1, ref2]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        r = self.get('/limits', expected_status=http.client.OK, auth=self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], domain_id=self.domain_id))
        limits = r.result['limits']
        self.assertEqual(1, len(limits))
        self.assertEqual(self.domain_id, limits[0]['domain_id'])
        r = self.get('/limits', expected_status=http.client.OK, auth=self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], domain_id=self.domain_2_id))
        limits = r.result['limits']
        self.assertEqual(1, len(limits))
        self.assertEqual(self.domain_2_id, limits[0]['domain_id'])
        r = self.get('/limits?domain_id=%s' % self.domain_id, expected_status=http.client.OK)
        limits = r.result['limits']
        self.assertEqual(0, len(limits))
        r = self.get('/limits?domain_id=%s' % self.domain_id, expected_status=http.client.OK, auth=self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], system=True))
        limits = r.result['limits']
        self.assertEqual(1, len(limits))
        self.assertEqual(self.domain_id, limits[0]['domain_id'])

    def test_show_project_limit(self):
        ref1 = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume')
        ref2 = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id2, resource_name='snapshot')
        r = self.post('/limits', body={'limits': [ref1, ref2]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        if r.result['limits'][0]['resource_name'] == 'volume':
            id1 = r.result['limits'][0]['id']
        else:
            id1 = r.result['limits'][1]['id']
        self.get('/limits/fake_id', token=self.system_admin_token, expected_status=http.client.NOT_FOUND)
        r = self.get('/limits/%s' % id1, expected_status=http.client.OK)
        limit = r.result['limit']
        self.assertIsNone(limit['domain_id'])
        for key in ['service_id', 'region_id', 'resource_name', 'resource_limit', 'description', 'project_id']:
            self.assertEqual(limit[key], ref1[key])

    def test_show_domain_limit(self):
        ref1 = unit.new_limit_ref(domain_id=self.domain_id, service_id=self.service_id2, resource_name='snapshot')
        r = self.post('/limits', body={'limits': [ref1]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        id1 = r.result['limits'][0]['id']
        r = self.get('/limits/%s' % id1, expected_status=http.client.OK, auth=self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], domain_id=self.domain_id))
        limit = r.result['limit']
        self.assertIsNone(limit['project_id'])
        self.assertIsNone(limit['region_id'])
        for key in ['service_id', 'resource_name', 'resource_limit', 'description', 'domain_id']:
            self.assertEqual(limit[key], ref1[key])

    def test_delete_limit(self):
        ref1 = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume')
        ref2 = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id2, resource_name='snapshot')
        r = self.post('/limits', body={'limits': [ref1, ref2]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        id1 = r.result['limits'][0]['id']
        self.delete('/limits/%s' % id1, token=self.system_admin_token, expected_status=http.client.NO_CONTENT)
        self.delete('/limits/fake_id', token=self.system_admin_token, expected_status=http.client.NOT_FOUND)
        r = self.get('/limits', token=self.system_admin_token, expected_status=http.client.OK)
        limits = r.result['limits']
        self.assertEqual(len(limits), 1)