import copy
import http.client
import uuid
from oslo_serialization import jsonutils
from keystone.common.policies import role_assignment as rp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class ProjectAdminTests(base_classes.TestCaseWithBootstrap, common_auth.AuthTestMixin, _AssignmentTestUtilities, _ProjectUserTests):

    def setUp(self):
        super(ProjectAdminTests, self).setUp()
        self.loadapp()
        self.policy_file = self.useFixture(temporaryfile.SecureTempFile())
        self.policy_file_name = self.policy_file.file_name
        self.useFixture(ksfixtures.Policy(self.config_fixture, policy_file=self.policy_file_name))
        self._override_policy()
        self.config_fixture.config(group='oslo_policy', enforce_scope=True)
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        self.domain_id = domain['id']
        self.user_id = self.bootstrapper.admin_user_id
        project = unit.new_project_ref(domain_id=self.domain_id)
        project = PROVIDERS.resource_api.create_project(project['id'], project)
        self.project_id = project['id']
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.admin_role_id, user_id=self.user_id, project_id=self.project_id)
        self.expected = [{'user_id': self.user_id, 'project_id': self.project_id, 'role_id': self.bootstrapper.admin_role_id}]
        auth = self.build_authentication_request(user_id=self.user_id, password=self.bootstrapper.admin_password, project_id=self.project_id)
        with self.test_client() as c:
            r = c.post('/v3/auth/tokens', json=auth)
            self.token_id = r.headers['X-Subject-Token']
            self.headers = {'X-Auth-Token': self.token_id}

    def _override_policy(self):
        with open(self.policy_file_name, 'w') as f:
            overridden_policies = {'identity:list_role_assignments': rp.SYSTEM_READER_OR_DOMAIN_READER, 'identity:list_role_assignments_for_tree': rp.SYSTEM_READER_OR_PROJECT_DOMAIN_READER_OR_PROJECT_ADMIN}
            f.write(jsonutils.dumps(overridden_policies))

    def test_user_can_list_assignments_for_subtree_on_own_project(self):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=self.domain_id))
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=self.domain_id, parent_id=self.project_id))
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, user_id=user['id'], project_id=project['id'])
        expected = copy.copy(self.expected)
        expected.append({'project_id': project['id'], 'user_id': user['id'], 'role_id': self.bootstrapper.reader_role_id})
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?scope.project.id=%s&include_subtree' % self.project_id, headers=self.headers)
            self.assertEqual(len(expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, expected)

    def test_user_cannot_list_assignments_for_subtree_on_other_project(self):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=self.domain_id))
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=self.domain_id))
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, user_id=user['id'], project_id=project['id'])
        with self.test_client() as c:
            c.get('/v3/role_assignments?scope.project.id=%s&include_subtree' % project['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)