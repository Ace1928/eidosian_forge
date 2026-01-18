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
class ProjectUserTestsWithoutEnforceScope(base_classes.TestCaseWithBootstrap, common_auth.AuthTestMixin, _DomainAndProjectUserProjectEndpointTests, _SystemReaderAndMemberProjectEndpointTests):

    def _override_policy(self):
        with open(self.policy_file_name, 'w') as f:
            overridden_policies = {'identity:list_projects_for_endpoint': bp.SYSTEM_READER, 'identity:add_endpoint_to_project': bp.SYSTEM_ADMIN, 'identity:check_endpoint_in_project': bp.SYSTEM_READER, 'identity:list_endpoints_for_project': bp.SYSTEM_READER, 'identity:remove_endpoint_from_project': bp.SYSTEM_ADMIN}
            f.write(jsonutils.dumps(overridden_policies))

    def setUp(self):
        super(ProjectUserTestsWithoutEnforceScope, self).setUp()
        self.loadapp()
        self.policy_file = self.useFixture(temporaryfile.SecureTempFile())
        self.policy_file_name = self.policy_file.file_name
        self.useFixture(ksfixtures.Policy(self.config_fixture, policy_file=self.policy_file_name))
        self._override_policy()
        self.config_fixture.config(group='oslo_policy', enforce_scope=False)
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        user = unit.new_user_ref(domain_id=domain['id'])
        self.user_id = PROVIDERS.identity_api.create_user(user)['id']
        self.project_id = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=domain['id']))['id']
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.admin_role_id, user_id=self.user_id, project_id=self.project_id)
        auth = self.build_authentication_request(user_id=self.user_id, password=user['password'], project_id=self.project_id)
        with self.test_client() as c:
            r = c.post('/v3/auth/tokens', json=auth)
            self.token_id = r.headers['X-Subject-Token']
            self.headers = {'X-Auth-Token': self.token_id}