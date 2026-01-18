import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
class EndpointGroupCRUDTestCase(EndpointFilterTestCase):
    DEFAULT_ENDPOINT_GROUP_BODY = {'endpoint_group': {'description': 'endpoint group description', 'filters': {'interface': 'admin'}, 'name': 'endpoint_group_name'}}
    DEFAULT_ENDPOINT_GROUP_URL = '/OS-EP-FILTER/endpoint_groups'

    def test_create_endpoint_group(self):
        """POST /OS-EP-FILTER/endpoint_groups.

        Valid endpoint group test case.

        """
        r = self.post(self.DEFAULT_ENDPOINT_GROUP_URL, body=self.DEFAULT_ENDPOINT_GROUP_BODY)
        expected_filters = self.DEFAULT_ENDPOINT_GROUP_BODY['endpoint_group']['filters']
        expected_name = self.DEFAULT_ENDPOINT_GROUP_BODY['endpoint_group']['name']
        self.assertEqual(expected_filters, r.result['endpoint_group']['filters'])
        self.assertEqual(expected_name, r.result['endpoint_group']['name'])
        self.assertThat(r.result['endpoint_group']['links']['self'], matchers.EndsWith('/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s' % {'endpoint_group_id': r.result['endpoint_group']['id']}))

    def test_create_invalid_endpoint_group(self):
        """POST /OS-EP-FILTER/endpoint_groups.

        Invalid endpoint group creation test case.

        """
        invalid_body = copy.deepcopy(self.DEFAULT_ENDPOINT_GROUP_BODY)
        invalid_body['endpoint_group']['filters'] = {'foobar': 'admin'}
        self.post(self.DEFAULT_ENDPOINT_GROUP_URL, body=invalid_body, expected_status=http.client.BAD_REQUEST)

    def test_get_endpoint_group(self):
        """GET /OS-EP-FILTER/endpoint_groups/{endpoint_group}.

        Valid endpoint group test case.

        """
        response = self.post(self.DEFAULT_ENDPOINT_GROUP_URL, body=self.DEFAULT_ENDPOINT_GROUP_BODY)
        endpoint_group_id = response.result['endpoint_group']['id']
        endpoint_group_filters = response.result['endpoint_group']['filters']
        endpoint_group_name = response.result['endpoint_group']['name']
        url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s' % {'endpoint_group_id': endpoint_group_id}
        self.get(url)
        self.assertEqual(endpoint_group_id, response.result['endpoint_group']['id'])
        self.assertEqual(endpoint_group_filters, response.result['endpoint_group']['filters'])
        self.assertEqual(endpoint_group_name, response.result['endpoint_group']['name'])
        self.assertThat(response.result['endpoint_group']['links']['self'], matchers.EndsWith(url))

    def test_get_invalid_endpoint_group(self):
        """GET /OS-EP-FILTER/endpoint_groups/{endpoint_group}.

        Invalid endpoint group test case.

        """
        endpoint_group_id = 'foobar'
        url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s' % {'endpoint_group_id': endpoint_group_id}
        self.get(url, expected_status=http.client.NOT_FOUND)

    def test_check_endpoint_group(self):
        """HEAD /OS-EP-FILTER/endpoint_groups/{endpoint_group_id}.

        Valid endpoint_group_id test case.

        """
        endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
        url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s' % {'endpoint_group_id': endpoint_group_id}
        self.head(url, expected_status=http.client.OK)

    def test_check_invalid_endpoint_group(self):
        """HEAD /OS-EP-FILTER/endpoint_groups/{endpoint_group_id}.

        Invalid endpoint_group_id test case.

        """
        endpoint_group_id = 'foobar'
        url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s' % {'endpoint_group_id': endpoint_group_id}
        self.head(url, expected_status=http.client.NOT_FOUND)

    def test_patch_endpoint_group(self):
        """PATCH /OS-EP-FILTER/endpoint_groups/{endpoint_group}.

        Valid endpoint group patch test case.

        """
        body = copy.deepcopy(self.DEFAULT_ENDPOINT_GROUP_BODY)
        body['endpoint_group']['filters'] = {'region_id': 'UK'}
        body['endpoint_group']['name'] = 'patch_test'
        endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
        url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s' % {'endpoint_group_id': endpoint_group_id}
        r = self.patch(url, body=body)
        self.assertEqual(endpoint_group_id, r.result['endpoint_group']['id'])
        self.assertEqual(body['endpoint_group']['filters'], r.result['endpoint_group']['filters'])
        self.assertThat(r.result['endpoint_group']['links']['self'], matchers.EndsWith(url))

    def test_patch_nonexistent_endpoint_group(self):
        """PATCH /OS-EP-FILTER/endpoint_groups/{endpoint_group}.

        Invalid endpoint group patch test case.

        """
        body = {'endpoint_group': {'name': 'patch_test'}}
        url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s' % {'endpoint_group_id': 'ABC'}
        self.patch(url, body=body, expected_status=http.client.NOT_FOUND)

    def test_patch_invalid_endpoint_group(self):
        """PATCH /OS-EP-FILTER/endpoint_groups/{endpoint_group}.

        Valid endpoint group patch test case.

        """
        body = {'endpoint_group': {'description': 'endpoint group description', 'filters': {'region': 'UK'}, 'name': 'patch_test'}}
        endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
        url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s' % {'endpoint_group_id': endpoint_group_id}
        self.patch(url, body=body, expected_status=http.client.BAD_REQUEST)
        url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s' % {'endpoint_group_id': endpoint_group_id}
        r = self.get(url)
        del r.result['endpoint_group']['id']
        del r.result['endpoint_group']['links']
        self.assertDictEqual(self.DEFAULT_ENDPOINT_GROUP_BODY, r.result)

    def test_delete_endpoint_group(self):
        """GET /OS-EP-FILTER/endpoint_groups/{endpoint_group}.

        Valid endpoint group test case.

        """
        endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
        url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s' % {'endpoint_group_id': endpoint_group_id}
        self.delete(url)
        self.get(url, expected_status=http.client.NOT_FOUND)

    def test_delete_invalid_endpoint_group(self):
        """GET /OS-EP-FILTER/endpoint_groups/{endpoint_group}.

        Invalid endpoint group test case.

        """
        endpoint_group_id = 'foobar'
        url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s' % {'endpoint_group_id': endpoint_group_id}
        self.delete(url, expected_status=http.client.NOT_FOUND)

    def test_add_endpoint_group_to_project(self):
        """Create a valid endpoint group and project association."""
        endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
        self._create_endpoint_group_project_association(endpoint_group_id, self.project_id)

    def test_add_endpoint_group_to_project_with_invalid_project_id(self):
        """Create an invalid endpoint group and project association."""
        endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
        project_id = uuid.uuid4().hex
        url = self._get_project_endpoint_group_url(endpoint_group_id, project_id)
        self.put(url, expected_status=http.client.NOT_FOUND)

    def test_get_endpoint_group_in_project(self):
        """Test retrieving project endpoint group association."""
        endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
        url = self._get_project_endpoint_group_url(endpoint_group_id, self.project_id)
        self.put(url)
        response = self.get(url)
        self.assertEqual(endpoint_group_id, response.result['project_endpoint_group']['endpoint_group_id'])
        self.assertEqual(self.project_id, response.result['project_endpoint_group']['project_id'])

    def test_get_invalid_endpoint_group_in_project(self):
        """Test retrieving project endpoint group association."""
        endpoint_group_id = uuid.uuid4().hex
        project_id = uuid.uuid4().hex
        url = self._get_project_endpoint_group_url(endpoint_group_id, project_id)
        self.get(url, expected_status=http.client.NOT_FOUND)

    def test_list_endpoint_groups_in_project(self):
        """GET & HEAD /OS-EP-FILTER/projects/{project_id}/endpoint_groups."""
        endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
        url = self._get_project_endpoint_group_url(endpoint_group_id, self.project_id)
        self.put(url)
        url = '/OS-EP-FILTER/projects/%(project_id)s/endpoint_groups' % {'project_id': self.project_id}
        response = self.get(url, expected_status=http.client.OK)
        self.assertEqual(endpoint_group_id, response.result['endpoint_groups'][0]['id'])
        self.head(url, expected_status=http.client.OK)

    def test_list_endpoint_groups_in_invalid_project(self):
        """Test retrieving from invalid project."""
        project_id = uuid.uuid4().hex
        url = '/OS-EP-FILTER/projects/%(project_id)s/endpoint_groups' % {'project_id': project_id}
        self.get(url, expected_status=http.client.NOT_FOUND)
        self.head(url, expected_status=http.client.NOT_FOUND)

    def test_empty_endpoint_groups_in_project(self):
        """Test when no endpoint groups associated with the project."""
        url = '/OS-EP-FILTER/projects/%(project_id)s/endpoint_groups' % {'project_id': self.project_id}
        response = self.get(url, expected_status=http.client.OK)
        self.assertEqual(0, len(response.result['endpoint_groups']))
        self.head(url, expected_status=http.client.OK)

    def test_check_endpoint_group_to_project(self):
        """Test HEAD with a valid endpoint group and project association."""
        endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
        self._create_endpoint_group_project_association(endpoint_group_id, self.project_id)
        url = self._get_project_endpoint_group_url(endpoint_group_id, self.project_id)
        self.head(url, expected_status=http.client.OK)

    def test_check_endpoint_group_to_project_with_invalid_project_id(self):
        """Test HEAD with an invalid endpoint group and project association."""
        endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
        url = self._get_project_endpoint_group_url(endpoint_group_id, self.project_id)
        self.put(url)
        project_id = uuid.uuid4().hex
        url = self._get_project_endpoint_group_url(endpoint_group_id, project_id)
        self.head(url, expected_status=http.client.NOT_FOUND)

    def test_list_endpoint_groups(self):
        """GET & HEAD /OS-EP-FILTER/endpoint_groups."""
        endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
        url = '/OS-EP-FILTER/endpoint_groups'
        r = self.get(url, expected_status=http.client.OK)
        self.assertNotEmpty(r.result['endpoint_groups'])
        self.assertEqual(endpoint_group_id, r.result['endpoint_groups'][0].get('id'))
        self.head(url, expected_status=http.client.OK)

    def test_list_endpoint_groups_by_name(self):
        """GET & HEAD /OS-EP-FILTER/endpoint_groups."""
        endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
        url = '/OS-EP-FILTER/endpoint_groups?name=%(name)s' % {'name': 'endpoint_group_name'}
        r = self.get(url, expected_status=http.client.OK)
        self.assertNotEmpty(r.result['endpoint_groups'])
        self.assertEqual(1, len(r.result['endpoint_groups']))
        self.assertEqual(endpoint_group_id, r.result['endpoint_groups'][0].get('id'))
        self.head(url, expected_status=http.client.OK)
        url = '/OS-EP-FILTER/endpoint_groups?name=%(name)s' % {'name': 'fake'}
        r = self.get(url, expected_status=http.client.OK)
        self.assertEqual(0, len(r.result['endpoint_groups']))

    def test_list_projects_associated_with_endpoint_group(self):
        """GET & HEAD /OS-EP-FILTER/endpoint_groups/{endpoint_group}/projects.

        Valid endpoint group test case.

        """
        endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
        self._create_endpoint_group_project_association(endpoint_group_id, self.project_id)
        url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s/projects' % {'endpoint_group_id': endpoint_group_id}
        self.get(url, expected_status=http.client.OK)
        self.head(url, expected_status=http.client.OK)

    def test_list_endpoints_associated_with_endpoint_group(self):
        """GET & HEAD /OS-EP-FILTER/endpoint_groups/{endpoint_group}/endpoints.

        Valid endpoint group test case.

        """
        service_ref = unit.new_service_ref()
        response = self.post('/services', body={'service': service_ref})
        service_id = response.result['service']['id']
        endpoint_ref = unit.new_endpoint_ref(service_id=service_id, interface='public', region_id=self.region_id)
        response = self.post('/endpoints', body={'endpoint': endpoint_ref})
        endpoint_id = response.result['endpoint']['id']
        body = copy.deepcopy(self.DEFAULT_ENDPOINT_GROUP_BODY)
        body['endpoint_group']['filters'] = {'service_id': service_id}
        endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, body)
        self._create_endpoint_group_project_association(endpoint_group_id, self.project_id)
        url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s/endpoints' % {'endpoint_group_id': endpoint_group_id}
        r = self.get(url, expected_status=http.client.OK)
        self.assertNotEmpty(r.result['endpoints'])
        self.assertEqual(endpoint_id, r.result['endpoints'][0].get('id'))
        self.head(url, expected_status=http.client.OK)

    def test_list_endpoints_associated_with_project_endpoint_group(self):
        """GET & HEAD /OS-EP-FILTER/projects/{project_id}/endpoints.

        Valid project, endpoint id, and endpoint group test case.

        """
        service_ref = unit.new_service_ref()
        response = self.post('/services', body={'service': service_ref})
        service_id2 = response.result['service']['id']
        self._create_endpoint_and_associations(self.default_domain_project_id, service_id2)
        self._create_endpoint_and_associations(self.default_domain_project_id)
        self.put(self.default_request_url)
        body = copy.deepcopy(self.DEFAULT_ENDPOINT_GROUP_BODY)
        body['endpoint_group']['filters'] = {'service_id': service_id2}
        endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, body)
        self._create_endpoint_group_project_association(endpoint_group_id, self.default_domain_project_id)
        endpoints_url = '/OS-EP-FILTER/projects/%(project_id)s/endpoints' % {'project_id': self.default_domain_project_id}
        r = self.get(endpoints_url, expected_status=http.client.OK)
        endpoints = self.assertValidEndpointListResponse(r)
        self.assertEqual(2, len(endpoints))
        self.head(endpoints_url, expected_status=http.client.OK)
        user_id = uuid.uuid4().hex
        catalog_list = PROVIDERS.catalog_api.get_v3_catalog(user_id, self.default_domain_project_id)
        self.assertEqual(2, len(catalog_list))
        url = self._get_project_endpoint_group_url(endpoint_group_id, self.default_domain_project_id)
        self.delete(url)
        url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s' % {'endpoint_group_id': endpoint_group_id}
        self.delete(url)
        r = self.get(endpoints_url)
        endpoints = self.assertValidEndpointListResponse(r)
        self.assertEqual(1, len(endpoints))
        catalog_list = PROVIDERS.catalog_api.get_v3_catalog(user_id, self.default_domain_project_id)
        self.assertEqual(1, len(catalog_list))

    def test_endpoint_group_project_cleanup_with_project(self):
        endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
        project_ref = unit.new_project_ref(domain_id=self.domain_id)
        r = self.post('/projects', body={'project': project_ref})
        project = self.assertValidProjectResponse(r, project_ref)
        url = self._get_project_endpoint_group_url(endpoint_group_id, project['id'])
        self.put(url)
        self.get(url, expected_status=http.client.OK)
        self.get(url, expected_status=http.client.OK)
        self.delete('/projects/%(project_id)s' % {'project_id': project['id']})
        self.get(url, expected_status=http.client.NOT_FOUND)
        self.head(url, expected_status=http.client.NOT_FOUND)

    def test_endpoint_group_project_cleanup_with_endpoint_group(self):
        endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
        project_ref = unit.new_project_ref(domain_id=self.domain_id)
        r = self.post('/projects', body={'project': project_ref})
        project = self.assertValidProjectResponse(r, project_ref)
        url = self._get_project_endpoint_group_url(endpoint_group_id, project['id'])
        self.put(url)
        self.get(url)
        self.delete(url)
        self.get(url, expected_status=http.client.NOT_FOUND)

    def test_removing_an_endpoint_group_project(self):
        endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
        url = self._get_project_endpoint_group_url(endpoint_group_id, self.default_domain_project_id)
        self.put(url)
        self.delete(url)
        self.get(url, expected_status=http.client.NOT_FOUND)

    def test_remove_endpoint_group_with_project_association(self):
        endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
        project_endpoint_group_url = self._get_project_endpoint_group_url(endpoint_group_id, self.default_domain_project_id)
        self.put(project_endpoint_group_url)
        endpoint_group_url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s' % {'endpoint_group_id': endpoint_group_id}
        self.delete(endpoint_group_url)
        self.get(endpoint_group_url, expected_status=http.client.NOT_FOUND)
        self.get(project_endpoint_group_url, expected_status=http.client.NOT_FOUND)

    @unit.skip_if_cache_disabled('catalog')
    def test_add_endpoint_group_to_project_invalidates_catalog_cache(self):
        endpoint_id2 = uuid.uuid4().hex
        endpoint2 = unit.new_endpoint_ref(service_id=self.service_id, region_id=self.region_id, interface='admin', id=endpoint_id2)
        PROVIDERS.catalog_api.create_endpoint(endpoint_id2, endpoint2)
        self.put(self.default_request_url)
        user_id = uuid.uuid4().hex
        catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, self.default_domain_project_id)
        self.assertThat(catalog[0]['endpoints'], matchers.HasLength(1))
        endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
        PROVIDERS.catalog_api.driver.add_endpoint_group_to_project(endpoint_group_id, self.default_domain_project_id)
        invalid_catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, self.default_domain_project_id)
        self.assertThat(invalid_catalog[0]['endpoints'], matchers.HasLength(1))
        self.assertEqual(catalog, invalid_catalog)
        PROVIDERS.catalog_api.driver.remove_endpoint_group_from_project(endpoint_group_id, self.default_domain_project_id)
        PROVIDERS.catalog_api.add_endpoint_group_to_project(endpoint_group_id, self.default_domain_project_id)
        catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, self.default_domain_project_id)
        self.assertThat(catalog[0]['endpoints'], matchers.HasLength(2))
        ep_id_list = [catalog[0]['endpoints'][0]['id'], catalog[0]['endpoints'][1]['id']]
        self.assertCountEqual([self.endpoint_id, endpoint_id2], ep_id_list)

    @unit.skip_if_cache_disabled('catalog')
    def test_remove_endpoint_group_from_project_invalidates_cache(self):
        endpoint_id2 = uuid.uuid4().hex
        endpoint2 = unit.new_endpoint_ref(service_id=self.service_id, region_id=self.region_id, interface='admin', id=endpoint_id2)
        PROVIDERS.catalog_api.create_endpoint(endpoint_id2, endpoint2)
        self.put(self.default_request_url)
        endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
        PROVIDERS.catalog_api.add_endpoint_group_to_project(endpoint_group_id, self.default_domain_project_id)
        user_id = uuid.uuid4().hex
        catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, self.default_domain_project_id)
        self.assertThat(catalog[0]['endpoints'], matchers.HasLength(2))
        ep_id_list = [catalog[0]['endpoints'][0]['id'], catalog[0]['endpoints'][1]['id']]
        self.assertCountEqual([self.endpoint_id, endpoint_id2], ep_id_list)
        PROVIDERS.catalog_api.driver.remove_endpoint_group_from_project(endpoint_group_id, self.default_domain_project_id)
        invalid_catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, self.default_domain_project_id)
        self.assertThat(invalid_catalog[0]['endpoints'], matchers.HasLength(2))
        self.assertEqual(catalog, invalid_catalog)
        PROVIDERS.catalog_api.driver.add_endpoint_group_to_project(endpoint_group_id, self.default_domain_project_id)
        PROVIDERS.catalog_api.remove_endpoint_group_from_project(endpoint_group_id, self.default_domain_project_id)
        catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, self.default_domain_project_id)
        self.assertThat(catalog[0]['endpoints'], matchers.HasLength(1))
        self.assertEqual(self.endpoint_id, catalog[0]['endpoints'][0]['id'])

    def _create_valid_endpoint_group(self, url, body):
        r = self.post(url, body=body)
        return r.result['endpoint_group']['id']

    def _create_endpoint_group_project_association(self, endpoint_group_id, project_id):
        url = self._get_project_endpoint_group_url(endpoint_group_id, project_id)
        self.put(url)

    def _get_project_endpoint_group_url(self, endpoint_group_id, project_id):
        return '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s/projects/%(project_id)s' % {'endpoint_group_id': endpoint_group_id, 'project_id': project_id}

    def _create_endpoint_and_associations(self, project_id, service_id=None):
        """Create an endpoint associated with service and project."""
        if not service_id:
            service_ref = unit.new_service_ref()
            response = self.post('/services', body={'service': service_ref})
            service_id = response.result['service']['id']
        endpoint_ref = unit.new_endpoint_ref(service_id=service_id, interface='public', region_id=self.region_id)
        response = self.post('/endpoints', body={'endpoint': endpoint_ref})
        endpoint = response.result['endpoint']
        self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.project['id'], 'endpoint_id': endpoint['id']})
        return endpoint