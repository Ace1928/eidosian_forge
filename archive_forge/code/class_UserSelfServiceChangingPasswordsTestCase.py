import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_db import exception as oslo_db_exception
from oslo_log import log
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import base as identity_base
from keystone.identity.backends import resource_options as options
from keystone.identity.backends import sql_model as model
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
class UserSelfServiceChangingPasswordsTestCase(ChangePasswordTestCase):

    def _create_user_with_expired_password(self):
        expire_days = CONF.security_compliance.password_expires_days + 1
        time = datetime.datetime.utcnow() - datetime.timedelta(expire_days)
        password = uuid.uuid4().hex
        user_ref = unit.new_user_ref(domain_id=self.domain_id, password=password)
        with freezegun.freeze_time(time):
            self.user_ref = PROVIDERS.identity_api.create_user(user_ref)
        return password

    def test_changing_password(self):
        token_id = self.get_request_token(self.user_ref['password'], expected_status=http.client.CREATED)
        old_token_auth = self.build_authentication_request(token=token_id)
        self.v3_create_token(old_token_auth)
        new_password = uuid.uuid4().hex
        self.change_password(password=new_password, original_password=self.user_ref['password'], expected_status=http.client.NO_CONTENT)
        self.get_request_token(self.user_ref['password'], expected_status=http.client.UNAUTHORIZED)
        self.v3_create_token(old_token_auth, expected_status=http.client.NOT_FOUND)
        self.get_request_token(new_password, expected_status=http.client.CREATED)

    def test_changing_password_with_min_password_age(self):
        time = datetime.datetime.utcnow()
        with freezegun.freeze_time(time) as frozen_datetime:
            new_password = uuid.uuid4().hex
            self.config_fixture.config(group='security_compliance', minimum_password_age=1)
            self.change_password(password=new_password, original_password=self.user_ref['password'], expected_status=http.client.NO_CONTENT)
            frozen_datetime.tick(delta=datetime.timedelta(seconds=1))
            self.token = self.get_request_token(new_password, http.client.CREATED)
            self.change_password(password=uuid.uuid4().hex, original_password=new_password, expected_status=http.client.BAD_REQUEST)
            self.config_fixture.config(group='security_compliance', minimum_password_age=0)
            self.change_password(password=uuid.uuid4().hex, original_password=new_password, expected_status=http.client.NO_CONTENT)

    def test_changing_password_with_password_lock(self):
        password = uuid.uuid4().hex
        ref = unit.new_user_ref(domain_id=self.domain_id, password=password)
        response = self.post('/users', body={'user': ref})
        user_id = response.json_body['user']['id']
        time = datetime.datetime.utcnow()
        with freezegun.freeze_time(time) as frozen_datetime:
            lock_pw_opt = options.LOCK_PASSWORD_OPT.option_name
            user_patch = {'user': {'options': {lock_pw_opt: True}}}
            self.patch('/users/%s' % user_id, body=user_patch)
            new_password = uuid.uuid4().hex
            body = {'user': {'original_password': password, 'password': new_password}}
            path = '/users/%s/password' % user_id
            self.post(path, body=body, expected_status=http.client.BAD_REQUEST)
            user_patch['user']['options'][lock_pw_opt] = False
            self.patch('/users/%s' % user_id, body=user_patch)
            path = '/users/%s/password' % user_id
            self.post(path, body=body, expected_status=http.client.NO_CONTENT)
            frozen_datetime.tick(delta=datetime.timedelta(seconds=1))
            auth_data = self.build_authentication_request(user_id=user_id, password=new_password)
            self.v3_create_token(auth_data, expected_status=http.client.CREATED)
            path = '/users/%s' % user_id
            user = self.get(path).json_body['user']
            self.assertIn(lock_pw_opt, user['options'])
            self.assertFalse(user['options'][lock_pw_opt])
            user_patch['user']['options'][lock_pw_opt] = None
            self.patch('/users/%s' % user_id, body=user_patch)
            path = '/users/%s' % user_id
            user = self.get(path).json_body['user']
            self.assertNotIn(lock_pw_opt, user['options'])

    def test_changing_password_with_missing_original_password_fails(self):
        r = self.change_password(password=uuid.uuid4().hex, expected_status=http.client.BAD_REQUEST)
        self.assertThat(r.result['error']['message'], matchers.Contains('original_password'))

    def test_changing_password_with_missing_password_fails(self):
        r = self.change_password(original_password=self.user_ref['password'], expected_status=http.client.BAD_REQUEST)
        self.assertThat(r.result['error']['message'], matchers.Contains('password'))

    def test_changing_password_with_incorrect_password_fails(self):
        self.change_password(password=uuid.uuid4().hex, original_password=uuid.uuid4().hex, expected_status=http.client.UNAUTHORIZED)

    def test_changing_password_with_disabled_user_fails(self):
        self.user_ref['enabled'] = False
        self.patch('/users/%s' % self.user_ref['id'], body={'user': self.user_ref})
        self.change_password(password=uuid.uuid4().hex, original_password=self.user_ref['password'], expected_status=http.client.UNAUTHORIZED)

    def test_changing_password_not_logged(self):
        log_fix = self.useFixture(fixtures.FakeLogger(level=log.DEBUG))
        new_password = uuid.uuid4().hex
        self.change_password(password=new_password, original_password=self.user_ref['password'], expected_status=http.client.NO_CONTENT)
        self.assertNotIn(self.user_ref['password'], log_fix.output)
        self.assertNotIn(new_password, log_fix.output)

    def test_changing_expired_password_succeeds(self):
        self.config_fixture.config(group='security_compliance', password_expires_days=2)
        password = self._create_user_with_expired_password()
        new_password = uuid.uuid4().hex
        self.change_password(password=new_password, original_password=password, expected_status=http.client.NO_CONTENT)
        self.get_request_token(new_password, expected_status=http.client.CREATED)

    def test_changing_expired_password_with_disabled_user_fails(self):
        self.config_fixture.config(group='security_compliance', password_expires_days=2)
        password = self._create_user_with_expired_password()
        self.user_ref['enabled'] = False
        self.patch('/users/%s' % self.user_ref['id'], body={'user': self.user_ref})
        new_password = uuid.uuid4().hex
        self.change_password(password=new_password, original_password=password, expected_status=http.client.UNAUTHORIZED)

    def test_change_password_required_upon_first_use_for_create(self):
        self.config_fixture.config(group='security_compliance', change_password_upon_first_use=True)
        self.user_ref = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
        self.get_request_token(self.user_ref['password'], expected_status=http.client.UNAUTHORIZED)
        new_password = uuid.uuid4().hex
        self.change_password(password=new_password, original_password=self.user_ref['password'], expected_status=http.client.NO_CONTENT)
        self.token = self.get_request_token(new_password, http.client.CREATED)

    def test_change_password_required_upon_first_use_for_admin_reset(self):
        self.config_fixture.config(group='security_compliance', change_password_upon_first_use=True)
        reset_password = uuid.uuid4().hex
        user_password = {'password': reset_password}
        PROVIDERS.identity_api.update_user(self.user_ref['id'], user_password)
        self.get_request_token(reset_password, expected_status=http.client.UNAUTHORIZED)
        new_password = uuid.uuid4().hex
        self.change_password(password=new_password, original_password=reset_password, expected_status=http.client.NO_CONTENT)
        self.token = self.get_request_token(new_password, http.client.CREATED)

    def test_change_password_required_upon_first_use_ignore_user(self):
        self.config_fixture.config(group='security_compliance', change_password_upon_first_use=True)
        reset_password = uuid.uuid4().hex
        self.user_ref['password'] = reset_password
        ignore_opt_name = options.IGNORE_CHANGE_PASSWORD_OPT.option_name
        self.user_ref['options'][ignore_opt_name] = True
        PROVIDERS.identity_api.update_user(self.user_ref['id'], self.user_ref)
        self.token = self.get_request_token(reset_password, http.client.CREATED)

    def test_lockout_exempt(self):
        self.config_fixture.config(group='security_compliance', lockout_failure_attempts=1)
        self.user_ref = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
        ignore_opt_name = options.IGNORE_LOCKOUT_ATTEMPT_OPT.option_name
        self.user_ref['options'][ignore_opt_name] = True
        PROVIDERS.identity_api.update_user(self.user_ref['id'], self.user_ref)
        bad_password = uuid.uuid4().hex
        self.token = self.get_request_token(bad_password, http.client.UNAUTHORIZED)
        self.get_request_token(self.user_ref['password'], expected_status=http.client.CREATED)