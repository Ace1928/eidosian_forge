import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class AdminTokenTests(TrustTests, _AdminTestsMixin):
    """Tests for the is_admin user.

    The Trusts API has hardcoded is_admin checks that we need to ensure are
    preserved through the system-scope transition.
    """

    def setUp(self):
        super(AdminTokenTests, self).setUp()
        self.config_fixture.config(admin_token='ADMIN')
        self.headers = {'X-Auth-Token': 'ADMIN'}

    def test_admin_can_delete_trust_for_other_user(self):
        ref = PROVIDERS.trust_api.create_trust(self.trust_id, **self.trust_data)
        with self.test_client() as c:
            c.delete('/v3/OS-TRUST/trusts/%s' % ref['id'], headers=self.headers, expected_status_code=http.client.NO_CONTENT)

    def test_admin_can_get_non_existent_trust_not_found(self):
        trust_id = uuid.uuid4().hex
        with self.test_client() as c:
            c.get('/v3/OS-TRUST/trusts/%s' % trust_id, headers=self.headers, expected_status_code=http.client.NOT_FOUND)

    def test_admin_cannot_get_trust_for_other_user(self):
        PROVIDERS.trust_api.create_trust(self.trust_id, **self.trust_data)
        with self.test_client() as c:
            c.get('/v3/OS-TRUST/trusts/%s' % self.trust_id, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_admin_cannot_list_trust_roles_for_other_user(self):
        PROVIDERS.trust_api.create_trust(self.trust_id, **self.trust_data)
        with self.test_client() as c:
            c.get('/v3/OS-TRUST/trusts/%s/roles' % self.trust_id, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_admin_cannot_get_trust_role_for_other_user(self):
        PROVIDERS.trust_api.create_trust(self.trust_id, **self.trust_data)
        with self.test_client() as c:
            c.get('/v3/OS-TRUST/trusts/%s/roles/%s' % (self.trust_id, self.bootstrapper.member_role_id), headers=self.headers, expected_status_code=http.client.FORBIDDEN)