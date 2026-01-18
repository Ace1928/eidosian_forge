import datetime
import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import base as base_policy
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class _SystemUserAndOwnerTests(object):
    """Common default functionality for all system users and owner."""

    def test_user_can_list_application_credentials(self):
        self._create_application_credential()
        self._create_application_credential()
        with self.test_client() as c:
            r = c.get('/v3/users/%s/application_credentials' % self.app_cred_user_id, headers=self.headers)
            self.assertEqual(2, len(r.json['application_credentials']))

    def test_user_can_get_application_credential(self):
        app_cred = self._create_application_credential()
        with self.test_client() as c:
            r = c.get('/v3/users/%s/application_credentials/%s' % (self.app_cred_user_id, app_cred['id']), headers=self.headers)
            actual_app_cred = r.json['application_credential']
            self.assertEqual(app_cred['id'], actual_app_cred['id'])

    def test_user_can_lookup_application_credential(self):
        app_cred = self._create_application_credential()
        with self.test_client() as c:
            r = c.get('/v3/users/%s/application_credentials?name=%s' % (self.app_cred_user_id, app_cred['name']), headers=self.headers)
            self.assertEqual(1, len(r.json['application_credentials']))
            actual_app_cred = r.json['application_credentials'][0]
            self.assertEqual(app_cred['id'], actual_app_cred['id'])

    def _test_delete_application_credential(self, expected_status_code=http.client.NO_CONTENT):
        app_cred = self._create_application_credential()
        with self.test_client() as c:
            c.delete('/v3/users/%s/application_credentials/%s' % (self.app_cred_user_id, app_cred['id']), expected_status_code=expected_status_code, headers=self.headers)

    def test_user_cannot_create_app_credential_for_another_user(self):
        another_user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        another_user_id = PROVIDERS.identity_api.create_user(another_user)['id']
        app_cred_body = {'application_credential': unit.new_application_credential_ref(roles=[{'id': self.bootstrapper.member_role_id}])}
        with self.test_client() as c:
            c.post('/v3/users/%s/application_credentials' % another_user_id, json=app_cred_body, expected_status_code=http.client.FORBIDDEN, headers=self.headers)