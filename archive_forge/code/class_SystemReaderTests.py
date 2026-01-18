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
class SystemReaderTests(base_classes.TestCaseWithBootstrap, common_auth.AuthTestMixin, _SystemUserProjectEndpointTests, _SystemReaderAndMemberProjectEndpointTests):

    def setUp(self):
        super(SystemReaderTests, self).setUp()
        self.loadapp()
        self.useFixture(ksfixtures.Policy(self.config_fixture))
        self.config_fixture.config(group='oslo_policy', enforce_scope=True)
        system_reader = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        self.user_id = PROVIDERS.identity_api.create_user(system_reader)['id']
        PROVIDERS.assignment_api.create_system_grant_for_user(self.user_id, self.bootstrapper.reader_role_id)
        auth = self.build_authentication_request(user_id=self.user_id, password=system_reader['password'], system=True)
        with self.test_client() as c:
            r = c.post('/v3/auth/tokens', json=auth)
            self.token_id = r.headers['X-Subject-Token']
            self.headers = {'X-Auth-Token': self.token_id}