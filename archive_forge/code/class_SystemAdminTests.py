import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import base as bp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class SystemAdminTests(base_classes.TestCaseWithBootstrap, common_auth.AuthTestMixin, _SystemUserProjectEndpointTests):

    def setUp(self):
        super(SystemAdminTests, self).setUp()
        self.loadapp()
        self.useFixture(ksfixtures.Policy(self.config_fixture))
        self.config_fixture.config(group='oslo_policy', enforce_scope=True)
        self.user_id = self.bootstrapper.admin_user_id
        auth = self.build_authentication_request(user_id=self.user_id, password=self.bootstrapper.admin_password, system=True)
        with self.test_client() as c:
            r = c.post('/v3/auth/tokens', json=auth)
            self.token_id = r.headers['X-Subject-Token']
            self.headers = {'X-Auth-Token': self.token_id}

    def test_user_can_add_endpoint_to_project(self):
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        endpoint = unit.new_endpoint_ref(service['id'], region_id=None)
        endpoint = PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
        with self.test_client() as c:
            c.put('/v3/OS-EP-FILTER/projects/%s/endpoints/%s' % (project['id'], endpoint['id']), headers=self.headers, expected_status_code=http.client.NO_CONTENT)

    def test_user_can_remove_endpoint_from_project(self):
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        endpoint = unit.new_endpoint_ref(service['id'], region_id=None)
        endpoint = PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
        PROVIDERS.catalog_api.add_endpoint_to_project(endpoint['id'], project['id'])
        with self.test_client() as c:
            c.delete('/v3/OS-EP-FILTER/projects/%s/endpoints/%s' % (project['id'], endpoint['id']), headers=self.headers, expected_status_code=http.client.NO_CONTENT)