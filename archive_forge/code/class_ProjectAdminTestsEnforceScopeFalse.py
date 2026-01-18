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
class ProjectAdminTestsEnforceScopeFalse(base_classes.TestCaseWithBootstrap, common_auth.AuthTestMixin, _UserCredentialTests, _SystemUserCredentialTests):

    def setUp(self):
        super(ProjectAdminTestsEnforceScopeFalse, self).setUp()
        self.loadapp()
        self.useFixture(ksfixtures.Policy(self.config_fixture))
        self.config_fixture.config(group='oslo_policy', enforce_scope=False)
        self.user_id = self.bootstrapper.admin_user_id
        auth = self.build_authentication_request(user_id=self.user_id, password=self.bootstrapper.admin_password, project_id=self.bootstrapper.project_id)
        with self.test_client() as c:
            r = c.post('/v3/auth/tokens', json=auth)
            self.token_id = r.headers['X-Subject-Token']
            self.headers = {'X-Auth-Token': self.token_id}