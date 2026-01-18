import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
class _SystemReaderAndMemberUserEndpointGroupsTests(object):
    """Common default functionality for system readers and system members."""

    def test_user_cannot_create_endpoint_groups(self):
        create = {'endpoint_group': {'id': uuid.uuid4().hex, 'description': uuid.uuid4().hex, 'filters': {'interface': 'public'}, 'name': uuid.uuid4().hex}}
        with self.test_client() as c:
            c.post('/v3/OS-EP-FILTER/endpoint_groups', json=create, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_update_endpoint_groups(self):
        endpoint_group = unit.new_endpoint_group_ref(filters={'interface': 'public'})
        endpoint_group = PROVIDERS.catalog_api.create_endpoint_group(endpoint_group['id'], endpoint_group)
        update = {'endpoint_group': {'filters': {'interface': 'internal'}}}
        with self.test_client() as c:
            c.patch('/v3/OS-EP-FILTER/endpoint_groups/%s' % endpoint_group['id'], json=update, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_endpoint_groups(self):
        endpoint_group = unit.new_endpoint_group_ref(filters={'interface': 'public'})
        endpoint_group = PROVIDERS.catalog_api.create_endpoint_group(endpoint_group['id'], endpoint_group)
        with self.test_client() as c:
            c.delete('/v3/OS-EP-FILTER/endpoint_groups/%s' % endpoint_group['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_add_endpoint_group_to_project(self):
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        endpoint_group = unit.new_endpoint_group_ref(filters={'interface': 'public'})
        endpoint_group = PROVIDERS.catalog_api.create_endpoint_group(endpoint_group['id'], endpoint_group)
        with self.test_client() as c:
            c.put('/v3/OS-EP-FILTER/endpoint_groups/%s/projects/%s' % (endpoint_group['id'], project['id']), headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_cannot_remove_endpoint_group_from_project(self):
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        endpoint_group = unit.new_endpoint_group_ref(filters={'interface': 'public'})
        endpoint_group = PROVIDERS.catalog_api.create_endpoint_group(endpoint_group['id'], endpoint_group)
        with self.test_client() as c:
            c.delete('/v3/OS-EP-FILTER/endpoint_groups/%s/projects/%s' % (endpoint_group['id'], project['id']), headers=self.headers, expected_status_code=http.client.FORBIDDEN)