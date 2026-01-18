import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
class CatalogTestCase(test_v3.RestfulTestCase):
    """Test service & endpoint CRUD."""

    def test_create_region_with_id(self):
        """Call ``PUT /regions/{region_id}`` w/o an ID in the request body."""
        ref = unit.new_region_ref()
        region_id = ref.pop('id')
        r = self.put('/regions/%s' % region_id, body={'region': ref}, expected_status=http.client.CREATED)
        self.assertValidRegionResponse(r, ref)
        self.assertEqual(region_id, r.json['region']['id'])

    def test_create_region_with_matching_ids(self):
        """Call ``PUT /regions/{region_id}`` with an ID in the request body."""
        ref = unit.new_region_ref()
        region_id = ref['id']
        r = self.put('/regions/%s' % region_id, body={'region': ref}, expected_status=http.client.CREATED)
        self.assertValidRegionResponse(r, ref)
        self.assertEqual(region_id, r.json['region']['id'])

    def test_create_region_with_duplicate_id(self):
        """Call ``PUT /regions/{region_id}``."""
        ref = unit.new_region_ref()
        region_id = ref['id']
        self.put('/regions/%s' % region_id, body={'region': ref}, expected_status=http.client.CREATED)
        self.put('/regions/%s' % region_id, body={'region': ref}, expected_status=http.client.CONFLICT)

    def test_create_region(self):
        """Call ``POST /regions`` with an ID in the request body."""
        ref = unit.new_region_ref()
        r = self.post('/regions', body={'region': ref})
        self.assertValidRegionResponse(r, ref)
        r = self.get('/regions/%(region_id)s' % {'region_id': ref['id']})
        self.assertValidRegionResponse(r, ref)

    def test_create_region_with_empty_id(self):
        """Call ``POST /regions`` with an empty ID in the request body."""
        ref = unit.new_region_ref(id='')
        r = self.post('/regions', body={'region': ref})
        self.assertValidRegionResponse(r, ref)
        self.assertNotEmpty(r.result['region'].get('id'))

    def test_create_region_without_id(self):
        """Call ``POST /regions`` without an ID in the request body."""
        ref = unit.new_region_ref()
        del ref['id']
        r = self.post('/regions', body={'region': ref})
        self.assertValidRegionResponse(r, ref)

    def test_create_region_without_description(self):
        """Call ``POST /regions`` without description in the request body."""
        ref = unit.new_region_ref(description=None)
        del ref['description']
        r = self.post('/regions', body={'region': ref})
        ref['description'] = ''
        self.assertValidRegionResponse(r, ref)

    def test_create_regions_with_same_description_string(self):
        """Call ``POST /regions`` with duplicate descriptions."""
        region_desc = 'Some Region Description'
        ref1 = unit.new_region_ref(description=region_desc)
        ref2 = unit.new_region_ref(description=region_desc)
        resp1 = self.post('/regions', body={'region': ref1})
        self.assertValidRegionResponse(resp1, ref1)
        resp2 = self.post('/regions', body={'region': ref2})
        self.assertValidRegionResponse(resp2, ref2)

    def test_create_regions_without_descriptions(self):
        """Call ``POST /regions`` with no description."""
        ref1 = unit.new_region_ref()
        ref2 = unit.new_region_ref()
        del ref1['description']
        ref2['description'] = None
        resp1 = self.post('/regions', body={'region': ref1})
        resp2 = self.post('/regions', body={'region': ref2})
        ref1['description'] = ''
        ref2['description'] = ''
        self.assertValidRegionResponse(resp1, ref1)
        self.assertValidRegionResponse(resp2, ref2)

    def test_create_region_with_conflicting_ids(self):
        """Call ``PUT /regions/{region_id}`` with conflicting region IDs."""
        ref = unit.new_region_ref()
        self.put('/regions/%s' % uuid.uuid4().hex, body={'region': ref}, expected_status=http.client.BAD_REQUEST)

    def test_list_head_regions(self):
        """Call ``GET & HEAD /regions``."""
        resource_url = '/regions'
        r = self.get(resource_url)
        self.assertValidRegionListResponse(r, ref=self.region)
        self.head(resource_url, expected_status=http.client.OK)

    def _create_region_with_parent_id(self, parent_id=None):
        ref = unit.new_region_ref(parent_region_id=parent_id)
        return self.post('/regions', body={'region': ref})

    def test_list_regions_filtered_by_parent_region_id(self):
        """Call ``GET /regions?parent_region_id={parent_region_id}``."""
        new_region = self._create_region_with_parent_id()
        parent_id = new_region.result['region']['id']
        new_region = self._create_region_with_parent_id(parent_id)
        new_region = self._create_region_with_parent_id(parent_id)
        r = self.get('/regions?parent_region_id=%s' % parent_id)
        for region in r.result['regions']:
            self.assertEqual(parent_id, region['parent_region_id'])

    def test_get_head_region(self):
        """Call ``GET & HEAD /regions/{region_id}``."""
        resource_url = '/regions/%(region_id)s' % {'region_id': self.region_id}
        r = self.get(resource_url)
        self.assertValidRegionResponse(r, self.region)
        self.head(resource_url, expected_status=http.client.OK)

    def test_update_region(self):
        """Call ``PATCH /regions/{region_id}``."""
        region = unit.new_region_ref()
        del region['id']
        r = self.patch('/regions/%(region_id)s' % {'region_id': self.region_id}, body={'region': region})
        self.assertValidRegionResponse(r, region)

    def test_update_region_without_description_keeps_original(self):
        """Call ``PATCH /regions/{region_id}``."""
        region_ref = unit.new_region_ref()
        resp = self.post('/regions', body={'region': region_ref})
        region_updates = {'parent_region_id': self.region_id}
        resp = self.patch('/regions/%s' % region_ref['id'], body={'region': region_updates})
        self.assertEqual(region_ref['description'], resp.result['region']['description'])

    def test_update_region_with_null_description(self):
        """Call ``PATCH /regions/{region_id}``."""
        region = unit.new_region_ref(description=None)
        del region['id']
        r = self.patch('/regions/%(region_id)s' % {'region_id': self.region_id}, body={'region': region})
        region['description'] = ''
        self.assertValidRegionResponse(r, region)

    def test_delete_region(self):
        """Call ``DELETE /regions/{region_id}``."""
        ref = unit.new_region_ref()
        r = self.post('/regions', body={'region': ref})
        self.assertValidRegionResponse(r, ref)
        self.delete('/regions/%(region_id)s' % {'region_id': ref['id']})

    def test_create_service(self):
        """Call ``POST /services``."""
        ref = unit.new_service_ref()
        r = self.post('/services', body={'service': ref})
        self.assertValidServiceResponse(r, ref)

    def test_create_service_no_name(self):
        """Call ``POST /services``."""
        ref = unit.new_service_ref()
        del ref['name']
        r = self.post('/services', body={'service': ref})
        ref['name'] = ''
        self.assertValidServiceResponse(r, ref)

    def test_create_service_no_enabled(self):
        """Call ``POST /services``."""
        ref = unit.new_service_ref()
        del ref['enabled']
        r = self.post('/services', body={'service': ref})
        ref['enabled'] = True
        self.assertValidServiceResponse(r, ref)
        self.assertIs(True, r.result['service']['enabled'])

    def test_create_service_enabled_false(self):
        """Call ``POST /services``."""
        ref = unit.new_service_ref(enabled=False)
        r = self.post('/services', body={'service': ref})
        self.assertValidServiceResponse(r, ref)
        self.assertIs(False, r.result['service']['enabled'])

    def test_create_service_enabled_true(self):
        """Call ``POST /services``."""
        ref = unit.new_service_ref(enabled=True)
        r = self.post('/services', body={'service': ref})
        self.assertValidServiceResponse(r, ref)
        self.assertIs(True, r.result['service']['enabled'])

    def test_create_service_enabled_str_true(self):
        """Call ``POST /services``."""
        ref = unit.new_service_ref(enabled='True')
        self.post('/services', body={'service': ref}, expected_status=http.client.BAD_REQUEST)

    def test_create_service_enabled_str_false(self):
        """Call ``POST /services``."""
        ref = unit.new_service_ref(enabled='False')
        self.post('/services', body={'service': ref}, expected_status=http.client.BAD_REQUEST)

    def test_create_service_enabled_str_random(self):
        """Call ``POST /services``."""
        ref = unit.new_service_ref(enabled='puppies')
        self.post('/services', body={'service': ref}, expected_status=http.client.BAD_REQUEST)

    def test_list_head_services(self):
        """Call ``GET & HEAD /services``."""
        resource_url = '/services'
        r = self.get(resource_url)
        self.assertValidServiceListResponse(r, ref=self.service)
        self.head(resource_url, expected_status=http.client.OK)

    def _create_random_service(self):
        ref = unit.new_service_ref()
        response = self.post('/services', body={'service': ref})
        return response.json['service']

    def test_filter_list_services_by_type(self):
        """Call ``GET /services?type=<some type>``."""
        target_ref = self._create_random_service()
        self._create_random_service()
        self._create_random_service()
        response = self.get('/services?type=' + target_ref['type'])
        self.assertValidServiceListResponse(response, ref=target_ref)
        filtered_service_list = response.json['services']
        self.assertEqual(1, len(filtered_service_list))
        filtered_service = filtered_service_list[0]
        self.assertEqual(target_ref['type'], filtered_service['type'])

    def test_filter_list_services_by_name(self):
        """Call ``GET /services?name=<some name>``."""
        self._create_random_service()
        self._create_random_service()
        target_ref = self._create_random_service()
        response = self.get('/services?name=' + target_ref['name'])
        self.assertValidServiceListResponse(response, ref=target_ref)
        filtered_service_list = response.json['services']
        self.assertEqual(1, len(filtered_service_list))
        filtered_service = filtered_service_list[0]
        self.assertEqual(target_ref['name'], filtered_service['name'])

    def test_filter_list_services_by_name_with_list_limit(self):
        """Call ``GET /services?name=<some name>``."""
        self.config_fixture.config(list_limit=1)
        self.test_filter_list_services_by_name()

    def test_get_head_service(self):
        """Call ``GET & HEAD /services/{service_id}``."""
        resource_url = '/services/%(service_id)s' % {'service_id': self.service_id}
        r = self.get(resource_url)
        self.assertValidServiceResponse(r, self.service)
        self.head(resource_url, expected_status=http.client.OK)

    def test_update_service(self):
        """Call ``PATCH /services/{service_id}``."""
        service = unit.new_service_ref()
        del service['id']
        r = self.patch('/services/%(service_id)s' % {'service_id': self.service_id}, body={'service': service})
        self.assertValidServiceResponse(r, service)

    def test_delete_service(self):
        """Call ``DELETE /services/{service_id}``."""
        self.delete('/services/%(service_id)s' % {'service_id': self.service_id})

    def test_list_head_endpoints(self):
        """Call ``GET & HEAD /endpoints``."""
        resource_url = '/endpoints'
        r = self.get(resource_url)
        self.assertValidEndpointListResponse(r, ref=self.endpoint)
        self.head(resource_url, expected_status=http.client.OK)

    def _create_random_endpoint(self, interface='public', parent_region_id=None):
        region = self._create_region_with_parent_id(parent_id=parent_region_id)
        service = self._create_random_service()
        ref = unit.new_endpoint_ref(service_id=service['id'], interface=interface, region_id=region.result['region']['id'])
        response = self.post('/endpoints', body={'endpoint': ref})
        return response.json['endpoint']

    def test_list_endpoints_filtered_by_interface(self):
        """Call ``GET /endpoints?interface={interface}``."""
        ref = self._create_random_endpoint(interface='internal')
        response = self.get('/endpoints?interface=%s' % ref['interface'])
        self.assertValidEndpointListResponse(response, ref=ref)
        for endpoint in response.json['endpoints']:
            self.assertEqual(ref['interface'], endpoint['interface'])

    def test_list_endpoints_filtered_by_service_id(self):
        """Call ``GET /endpoints?service_id={service_id}``."""
        ref = self._create_random_endpoint()
        response = self.get('/endpoints?service_id=%s' % ref['service_id'])
        self.assertValidEndpointListResponse(response, ref=ref)
        for endpoint in response.json['endpoints']:
            self.assertEqual(ref['service_id'], endpoint['service_id'])

    def test_list_endpoints_filtered_by_region_id(self):
        """Call ``GET /endpoints?region_id={region_id}``."""
        ref = self._create_random_endpoint()
        response = self.get('/endpoints?region_id=%s' % ref['region_id'])
        self.assertValidEndpointListResponse(response, ref=ref)
        for endpoint in response.json['endpoints']:
            self.assertEqual(ref['region_id'], endpoint['region_id'])

    def test_list_endpoints_filtered_by_parent_region_id(self):
        """Call ``GET /endpoints?region_id={region_id}``.

        Ensure passing the parent_region_id as filter returns an
        empty list.

        """
        parent_region = self._create_region_with_parent_id()
        parent_region_id = parent_region.result['region']['id']
        self._create_random_endpoint(parent_region_id=parent_region_id)
        response = self.get('/endpoints?region_id=%s' % parent_region_id)
        self.assertEqual(0, len(response.json['endpoints']))

    def test_list_endpoints_with_multiple_filters(self):
        """Call ``GET /endpoints?interface={interface}...``.

        Ensure passing different combinations of interface, region_id and
        service_id as filters will return the correct result.

        """
        ref = self._create_random_endpoint(interface='internal')
        response = self.get('/endpoints?interface=%s&region_id=%s' % (ref['interface'], ref['region_id']))
        self.assertValidEndpointListResponse(response, ref=ref)
        for endpoint in response.json['endpoints']:
            self.assertEqual(ref['interface'], endpoint['interface'])
            self.assertEqual(ref['region_id'], endpoint['region_id'])
        ref = self._create_random_endpoint(interface='internal')
        response = self.get('/endpoints?interface=%s&service_id=%s' % (ref['interface'], ref['service_id']))
        self.assertValidEndpointListResponse(response, ref=ref)
        for endpoint in response.json['endpoints']:
            self.assertEqual(ref['interface'], endpoint['interface'])
            self.assertEqual(ref['service_id'], endpoint['service_id'])
        ref = self._create_random_endpoint(interface='internal')
        response = self.get('/endpoints?region_id=%s&service_id=%s' % (ref['region_id'], ref['service_id']))
        self.assertValidEndpointListResponse(response, ref=ref)
        for endpoint in response.json['endpoints']:
            self.assertEqual(ref['region_id'], endpoint['region_id'])
            self.assertEqual(ref['service_id'], endpoint['service_id'])
        ref = self._create_random_endpoint(interface='internal')
        response = self.get('/endpoints?interface=%s&region_id=%s&service_id=%s' % (ref['interface'], ref['region_id'], ref['service_id']))
        self.assertValidEndpointListResponse(response, ref=ref)
        for endpoint in response.json['endpoints']:
            self.assertEqual(ref['interface'], endpoint['interface'])
            self.assertEqual(ref['region_id'], endpoint['region_id'])
            self.assertEqual(ref['service_id'], endpoint['service_id'])

    def test_list_endpoints_with_random_filter_values(self):
        """Call ``GET /endpoints?interface={interface}...``.

        Ensure passing random values for: interface, region_id and
        service_id will return an empty list.

        """
        self._create_random_endpoint(interface='internal')
        response = self.get('/endpoints?interface=%s' % uuid.uuid4().hex)
        self.assertEqual(0, len(response.json['endpoints']))
        response = self.get('/endpoints?region_id=%s' % uuid.uuid4().hex)
        self.assertEqual(0, len(response.json['endpoints']))
        response = self.get('/endpoints?service_id=%s' % uuid.uuid4().hex)
        self.assertEqual(0, len(response.json['endpoints']))

    def test_create_endpoint_no_enabled(self):
        """Call ``POST /endpoints``."""
        ref = unit.new_endpoint_ref(service_id=self.service_id, interface='public', region_id=self.region_id)
        r = self.post('/endpoints', body={'endpoint': ref})
        ref['enabled'] = True
        self.assertValidEndpointResponse(r, ref)

    def test_create_endpoint_enabled_true(self):
        """Call ``POST /endpoints`` with enabled: true."""
        ref = unit.new_endpoint_ref(service_id=self.service_id, interface='public', region_id=self.region_id, enabled=True)
        r = self.post('/endpoints', body={'endpoint': ref})
        self.assertValidEndpointResponse(r, ref)

    def test_create_endpoint_enabled_false(self):
        """Call ``POST /endpoints`` with enabled: false."""
        ref = unit.new_endpoint_ref(service_id=self.service_id, interface='public', region_id=self.region_id, enabled=False)
        r = self.post('/endpoints', body={'endpoint': ref})
        self.assertValidEndpointResponse(r, ref)

    def test_create_endpoint_enabled_str_true(self):
        """Call ``POST /endpoints`` with enabled: 'True'."""
        ref = unit.new_endpoint_ref(service_id=self.service_id, interface='public', region_id=self.region_id, enabled='True')
        self.post('/endpoints', body={'endpoint': ref}, expected_status=http.client.BAD_REQUEST)

    def test_create_endpoint_enabled_str_false(self):
        """Call ``POST /endpoints`` with enabled: 'False'."""
        ref = unit.new_endpoint_ref(service_id=self.service_id, interface='public', region_id=self.region_id, enabled='False')
        self.post('/endpoints', body={'endpoint': ref}, expected_status=http.client.BAD_REQUEST)

    def test_create_endpoint_enabled_str_random(self):
        """Call ``POST /endpoints`` with enabled: 'puppies'."""
        ref = unit.new_endpoint_ref(service_id=self.service_id, interface='public', region_id=self.region_id, enabled='puppies')
        self.post('/endpoints', body={'endpoint': ref}, expected_status=http.client.BAD_REQUEST)

    def test_create_endpoint_with_invalid_region_id(self):
        """Call ``POST /endpoints``."""
        ref = unit.new_endpoint_ref(service_id=self.service_id)
        self.post('/endpoints', body={'endpoint': ref}, expected_status=http.client.BAD_REQUEST)

    def test_create_endpoint_with_region(self):
        """EndpointV3 creates the region before creating the endpoint.

        This occurs when endpoint is provided with 'region' and no 'region_id'.
        """
        ref = unit.new_endpoint_ref_with_region(service_id=self.service_id, region=uuid.uuid4().hex)
        self.post('/endpoints', body={'endpoint': ref})
        self.get('/regions/%(region_id)s' % {'region_id': ref['region']})

    def test_create_endpoint_with_no_region(self):
        """EndpointV3 allows to creates the endpoint without region."""
        ref = unit.new_endpoint_ref(service_id=self.service_id, region_id=None)
        del ref['region_id']
        self.post('/endpoints', body={'endpoint': ref})

    def test_create_endpoint_with_empty_url(self):
        """Call ``POST /endpoints``."""
        ref = unit.new_endpoint_ref(service_id=self.service_id, url='')
        self.post('/endpoints', body={'endpoint': ref}, expected_status=http.client.BAD_REQUEST)

    def test_get_head_endpoint(self):
        """Call ``GET & HEAD /endpoints/{endpoint_id}``."""
        resource_url = '/endpoints/%(endpoint_id)s' % {'endpoint_id': self.endpoint_id}
        r = self.get(resource_url)
        self.assertValidEndpointResponse(r, self.endpoint)
        self.head(resource_url, expected_status=http.client.OK)

    def test_update_endpoint(self):
        """Call ``PATCH /endpoints/{endpoint_id}``."""
        ref = unit.new_endpoint_ref(service_id=self.service_id, interface='public', region_id=self.region_id)
        del ref['id']
        r = self.patch('/endpoints/%(endpoint_id)s' % {'endpoint_id': self.endpoint_id}, body={'endpoint': ref})
        ref['enabled'] = True
        self.assertValidEndpointResponse(r, ref)

    def test_update_endpoint_enabled_true(self):
        """Call ``PATCH /endpoints/{endpoint_id}`` with enabled: True."""
        r = self.patch('/endpoints/%(endpoint_id)s' % {'endpoint_id': self.endpoint_id}, body={'endpoint': {'enabled': True}})
        self.assertValidEndpointResponse(r, self.endpoint)

    def test_update_endpoint_enabled_false(self):
        """Call ``PATCH /endpoints/{endpoint_id}`` with enabled: False."""
        r = self.patch('/endpoints/%(endpoint_id)s' % {'endpoint_id': self.endpoint_id}, body={'endpoint': {'enabled': False}})
        exp_endpoint = copy.copy(self.endpoint)
        exp_endpoint['enabled'] = False
        self.assertValidEndpointResponse(r, exp_endpoint)

    def test_update_endpoint_enabled_str_true(self):
        """Call ``PATCH /endpoints/{endpoint_id}`` with enabled: 'True'."""
        self.patch('/endpoints/%(endpoint_id)s' % {'endpoint_id': self.endpoint_id}, body={'endpoint': {'enabled': 'True'}}, expected_status=http.client.BAD_REQUEST)

    def test_update_endpoint_enabled_str_false(self):
        """Call ``PATCH /endpoints/{endpoint_id}`` with enabled: 'False'."""
        self.patch('/endpoints/%(endpoint_id)s' % {'endpoint_id': self.endpoint_id}, body={'endpoint': {'enabled': 'False'}}, expected_status=http.client.BAD_REQUEST)

    def test_update_endpoint_enabled_str_random(self):
        """Call ``PATCH /endpoints/{endpoint_id}`` with enabled: 'kitties'."""
        self.patch('/endpoints/%(endpoint_id)s' % {'endpoint_id': self.endpoint_id}, body={'endpoint': {'enabled': 'kitties'}}, expected_status=http.client.BAD_REQUEST)

    def test_delete_endpoint(self):
        """Call ``DELETE /endpoints/{endpoint_id}``."""
        self.delete('/endpoints/%(endpoint_id)s' % {'endpoint_id': self.endpoint_id})

    def test_deleting_endpoint_with_space_in_url(self):
        url_with_space = 'http://127.0.0.1:8774 /v1.1/\\$(tenant_i d)s'
        ref = unit.new_endpoint_ref(service_id=self.service['id'], region_id=None, publicurl=url_with_space, internalurl=url_with_space, adminurl=url_with_space, url=url_with_space)
        PROVIDERS.catalog_api.create_endpoint(ref['id'], ref)
        self.delete('/endpoints/%s' % ref['id'])
        self.get('/endpoints/%s' % ref['id'], expected_status=http.client.NOT_FOUND)

    def test_endpoint_create_with_valid_url(self):
        """Create endpoint with valid url should be tested,too."""
        valid_url = 'http://127.0.0.1:8774/v1.1/$(project_id)s'
        ref = unit.new_endpoint_ref(self.service_id, interface='public', region_id=self.region_id, url=valid_url)
        self.post('/endpoints', body={'endpoint': ref})

    def test_endpoint_create_with_valid_url_project_id(self):
        """Create endpoint with valid url should be tested,too."""
        valid_url = 'http://127.0.0.1:8774/v1.1/$(project_id)s'
        ref = unit.new_endpoint_ref(self.service_id, interface='public', region_id=self.region_id, url=valid_url)
        self.post('/endpoints', body={'endpoint': ref})

    def test_endpoint_create_with_invalid_url(self):
        """Test the invalid cases: substitutions is not exactly right."""
        invalid_urls = ['http://127.0.0.1:8774/v1.1/$(nonexistent)s', 'http://127.0.0.1:8774/v1.1/$(project_id)', 'http://127.0.0.1:8774/v1.1/$(project_id)t', 'http://127.0.0.1:8774/v1.1/$(project_id', 'http://127.0.0.1:8774/v1.1/$(admin_url)d']
        ref = unit.new_endpoint_ref(self.service_id)
        for invalid_url in invalid_urls:
            ref['url'] = invalid_url
            self.post('/endpoints', body={'endpoint': ref}, expected_status=http.client.BAD_REQUEST)