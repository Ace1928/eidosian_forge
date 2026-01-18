from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
class CloudServices(base.TestCase):

    def setUp(self, cloud_config_fixture='clouds.yaml'):
        super(CloudServices, self).setUp(cloud_config_fixture)

    def get_mock_url(self, service_type='identity', interface='public', resource='services', append=None, base_url_append='v3'):
        return super(CloudServices, self).get_mock_url(service_type, interface, resource, append, base_url_append)

    def test_create_service_v3(self):
        service_data = self._get_service_data(name='a service', type='network', description='A test service')
        self.register_uris([dict(method='POST', uri=self.get_mock_url(), status_code=200, json=service_data.json_response_v3, validate=dict(json={'service': service_data.json_request}))])
        service = self.cloud.create_service(name=service_data.service_name, service_type=service_data.service_type, description=service_data.description)
        self.assertThat(service.name, matchers.Equals(service_data.service_name))
        self.assertThat(service.id, matchers.Equals(service_data.service_id))
        self.assertThat(service.description, matchers.Equals(service_data.description))
        self.assertThat(service.type, matchers.Equals(service_data.service_type))
        self.assert_calls()

    def test_update_service_v3(self):
        service_data = self._get_service_data(name='a service', type='network', description='A test service')
        request = service_data.json_request.copy()
        request['enabled'] = False
        resp = service_data.json_response_v3.copy()
        resp['enabled'] = False
        request.pop('description')
        request.pop('name')
        request.pop('type')
        self.register_uris([dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'services': [resp['service']]}), dict(method='PATCH', uri=self.get_mock_url(append=[service_data.service_id]), status_code=200, json=resp, validate=dict(json={'service': request}))])
        service = self.cloud.update_service(service_data.service_id, enabled=False)
        self.assertThat(service.name, matchers.Equals(service_data.service_name))
        self.assertThat(service.id, matchers.Equals(service_data.service_id))
        self.assertThat(service.description, matchers.Equals(service_data.description))
        self.assertThat(service.type, matchers.Equals(service_data.service_type))
        self.assert_calls()

    def test_list_services(self):
        service_data = self._get_service_data()
        self.register_uris([dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'services': [service_data.json_response_v3['service']]})])
        services = self.cloud.list_services()
        self.assertThat(len(services), matchers.Equals(1))
        self.assertThat(services[0].id, matchers.Equals(service_data.service_id))
        self.assertThat(services[0].name, matchers.Equals(service_data.service_name))
        self.assertThat(services[0].type, matchers.Equals(service_data.service_type))
        self.assert_calls()

    def test_get_service(self):
        service_data = self._get_service_data()
        service2_data = self._get_service_data()
        self.register_uris([dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'services': [service_data.json_response_v3['service'], service2_data.json_response_v3['service']]}), dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'services': [service_data.json_response_v3['service'], service2_data.json_response_v3['service']]}), dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'services': [service_data.json_response_v3['service'], service2_data.json_response_v3['service']]}), dict(method='GET', uri=self.get_mock_url(), status_code=400)])
        service = self.cloud.get_service(name_or_id=service_data.service_id)
        self.assertThat(service.id, matchers.Equals(service_data.service_id))
        service = self.cloud.get_service(name_or_id=service_data.service_name)
        self.assertThat(service.id, matchers.Equals(service_data.service_id))
        service = self.cloud.get_service(name_or_id='INVALID SERVICE')
        self.assertIs(None, service)
        self.assertRaises(exceptions.SDKException, self.cloud.get_service, name_or_id=None, filters={'type': 'type2'})
        self.assert_calls()

    def test_search_services(self):
        service_data = self._get_service_data()
        service2_data = self._get_service_data(type=service_data.service_type)
        self.register_uris([dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'services': [service_data.json_response_v3['service'], service2_data.json_response_v3['service']]}), dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'services': [service_data.json_response_v3['service'], service2_data.json_response_v3['service']]}), dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'services': [service_data.json_response_v3['service'], service2_data.json_response_v3['service']]}), dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'services': [service_data.json_response_v3['service'], service2_data.json_response_v3['service']]})])
        services = self.cloud.search_services(name_or_id=service_data.service_id)
        self.assertThat(len(services), matchers.Equals(1))
        self.assertThat(services[0].id, matchers.Equals(service_data.service_id))
        services = self.cloud.search_services(name_or_id=service_data.service_name)
        self.assertThat(len(services), matchers.Equals(1))
        self.assertThat(services[0].name, matchers.Equals(service_data.service_name))
        services = self.cloud.search_services(name_or_id='!INVALID!')
        self.assertThat(len(services), matchers.Equals(0))
        services = self.cloud.search_services(filters={'type': service_data.service_type})
        self.assertThat(len(services), matchers.Equals(2))
        self.assertThat(services[0].id, matchers.Equals(service_data.service_id))
        self.assertThat(services[1].id, matchers.Equals(service2_data.service_id))
        self.assert_calls()

    def test_delete_service(self):
        service_data = self._get_service_data()
        self.register_uris([dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'services': [service_data.json_response_v3['service']]}), dict(method='DELETE', uri=self.get_mock_url(append=[service_data.service_id]), status_code=204), dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'services': [service_data.json_response_v3['service']]}), dict(method='DELETE', uri=self.get_mock_url(append=[service_data.service_id]), status_code=204)])
        self.cloud.delete_service(name_or_id=service_data.service_name)
        self.cloud.delete_service(service_data.service_id)
        self.assert_calls()