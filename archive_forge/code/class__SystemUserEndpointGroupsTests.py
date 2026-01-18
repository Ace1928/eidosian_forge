import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
class _SystemUserEndpointGroupsTests(object):
    """Common default functionality for all system users."""

    def test_user_can_list_endpoint_groups(self):
        endpoint_group = unit.new_endpoint_group_ref(filters={'interface': 'public'})
        endpoint_group = PROVIDERS.catalog_api.create_endpoint_group(endpoint_group['id'], endpoint_group)
        with self.test_client() as c:
            r = c.get('/v3/OS-EP-FILTER/endpoint_groups', headers=self.headers)
            endpoint_groups = []
            for endpoint_group in r.json['endpoint_groups']:
                endpoint_groups.append(endpoint_group['id'])
            self.assertIn(endpoint_group['id'], endpoint_groups)

    def test_user_can_get_an_endpoint_group(self):
        endpoint_group = unit.new_endpoint_group_ref(filters={'interface': 'public'})
        endpoint_group = PROVIDERS.catalog_api.create_endpoint_group(endpoint_group['id'], endpoint_group)
        with self.test_client() as c:
            c.get('/v3/OS-EP-FILTER/endpoint_groups/%s' % endpoint_group['id'], headers=self.headers)

    def test_user_can_list_projects_associated_with_endpoint_groups(self):
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        endpoint_group = unit.new_endpoint_group_ref(filters={'interface': 'public'})
        endpoint_group = PROVIDERS.catalog_api.create_endpoint_group(endpoint_group['id'], endpoint_group)
        PROVIDERS.catalog_api.add_endpoint_group_to_project(endpoint_group['id'], project['id'])
        with self.test_client() as c:
            r = c.get('/v3/OS-EP-FILTER/endpoint_groups/%s/projects' % endpoint_group['id'], headers=self.headers)
            projects = []
            for project in r.json['projects']:
                projects.append(project['id'])
            self.assertIn(project['id'], projects)

    def test_user_can_list_endpoints_associated_with_endpoint_groups(self):
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        endpoint = unit.new_endpoint_ref(service['id'], region_id=None)
        endpoint = PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
        endpoint_group = unit.new_endpoint_group_ref(filters={'interface': 'public'})
        endpoint_group = PROVIDERS.catalog_api.create_endpoint_group(endpoint_group['id'], endpoint_group)
        with self.test_client() as c:
            r = c.get('/v3/OS-EP-FILTER/endpoint_groups/%s/endpoints' % endpoint_group['id'], headers=self.headers)
            endpoints = []
            for endpoint in r.json['endpoints']:
                endpoints.append(endpoint['id'])
            self.assertIn(endpoint['id'], endpoints)

    def test_user_can_get_endpoints_associated_with_endpoint_groups(self):
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        endpoint_group = unit.new_endpoint_group_ref(filters={'interface': 'public'})
        endpoint_group = PROVIDERS.catalog_api.create_endpoint_group(endpoint_group['id'], endpoint_group)
        PROVIDERS.catalog_api.add_endpoint_group_to_project(endpoint_group['id'], project['id'])
        with self.test_client() as c:
            c.get('/v3/OS-EP-FILTER/endpoint_groups/%s/projects/%s' % (endpoint_group['id'], project['id']), headers=self.headers)

    def test_user_can_list_endpoint_groups_with_their_projects(self):
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        endpoint_group = unit.new_endpoint_group_ref(filters={'interface': 'public'})
        endpoint_group = PROVIDERS.catalog_api.create_endpoint_group(endpoint_group['id'], endpoint_group)
        PROVIDERS.catalog_api.add_endpoint_group_to_project(endpoint_group['id'], project['id'])
        with self.test_client() as c:
            r = c.get('/v3/OS-EP-FILTER/projects/%s/endpoint_groups' % project['id'], headers=self.headers)
            endpoint_groups = []
            for endpoint_group in r.json['endpoint_groups']:
                endpoint_groups.append(endpoint_group['id'])