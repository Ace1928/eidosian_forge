import copy
import datetime
import fixtures
import itertools
import operator
import re
from unittest import mock
from urllib import parse
import uuid
from cryptography.hazmat.primitives.serialization import Encoding
import freezegun
import http.client
from oslo_serialization import jsonutils as json
from oslo_utils import fixture
from oslo_utils import timeutils
from testtools import matchers
from testtools import testcase
from keystone import auth
from keystone.auth.plugins import totp
from keystone.common import authorization
from keystone.common import provider_api
from keystone.common.rbac_enforcer import policy
from keystone.common import utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import resource_options as ro
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
class TestAuthTOTP(test_v3.RestfulTestCase):

    def setUp(self):
        super(TestAuthTOTP, self).setUp()
        self.useFixture(ksfixtures.KeyRepository(self.config_fixture, 'credential', credential_fernet.MAX_ACTIVE_KEYS))
        ref = unit.new_totp_credential(user_id=self.default_domain_user['id'], project_id=self.default_domain_project['id'])
        self.secret = ref['blob']
        r = self.post('/credentials', body={'credential': ref})
        self.assertValidCredentialResponse(r, ref)
        self.addCleanup(self.cleanup)

    def auth_plugin_config_override(self):
        methods = ['totp', 'token', 'password']
        super(TestAuthTOTP, self).auth_plugin_config_override(methods)

    def _make_credentials(self, cred_type, count=1, user_id=None, project_id=None, blob=None):
        user_id = user_id or self.default_domain_user['id']
        project_id = project_id or self.default_domain_project['id']
        creds = []
        for __ in range(count):
            if cred_type == 'totp':
                ref = unit.new_totp_credential(user_id=user_id, project_id=project_id, blob=blob)
            else:
                ref = unit.new_credential_ref(user_id=user_id, project_id=project_id)
            resp = self.post('/credentials', body={'credential': ref})
            creds.append(resp.json['credential'])
        return creds

    def _make_auth_data_by_id(self, passcode, user_id=None):
        return self.build_authentication_request(user_id=user_id or self.default_domain_user['id'], passcode=passcode, project_id=self.project['id'])

    def _make_auth_data_by_name(self, passcode, username, user_domain_id):
        return self.build_authentication_request(username=username, user_domain_id=user_domain_id, passcode=passcode, project_id=self.project['id'])

    def cleanup(self):
        totp_creds = PROVIDERS.credential_api.list_credentials_for_user(self.default_domain_user['id'], type='totp')
        other_creds = PROVIDERS.credential_api.list_credentials_for_user(self.default_domain_user['id'], type='other')
        for cred in itertools.chain(other_creds, totp_creds):
            self.delete('/credentials/%s' % cred['id'], expected_status=http.client.NO_CONTENT)

    def test_with_a_valid_passcode(self):
        creds = self._make_credentials('totp')
        secret = creds[-1]['blob']
        self.useFixture(fixture.TimeFixture())
        auth_data = self._make_auth_data_by_id(totp._generate_totp_passcodes(secret)[0])
        self.v3_create_token(auth_data, expected_status=http.client.CREATED)

    def test_with_an_expired_passcode(self):
        creds = self._make_credentials('totp')
        secret = creds[-1]['blob']
        past = datetime.datetime.utcnow() - datetime.timedelta(minutes=2)
        with freezegun.freeze_time(past):
            auth_data = self._make_auth_data_by_id(totp._generate_totp_passcodes(secret)[0])
        self.useFixture(fixture.TimeFixture())
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_with_an_expired_passcode_no_previous_windows(self):
        self.config_fixture.config(group='totp', included_previous_windows=0)
        creds = self._make_credentials('totp')
        secret = creds[-1]['blob']
        past = datetime.datetime.utcnow() - datetime.timedelta(seconds=30)
        with freezegun.freeze_time(past):
            auth_data = self._make_auth_data_by_id(totp._generate_totp_passcodes(secret)[0])
        self.useFixture(fixture.TimeFixture())
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_with_passcode_no_previous_windows(self):
        self.config_fixture.config(group='totp', included_previous_windows=0)
        creds = self._make_credentials('totp')
        secret = creds[-1]['blob']
        auth_data = self._make_auth_data_by_id(totp._generate_totp_passcodes(secret)[0])
        self.useFixture(fixture.TimeFixture())
        self.v3_create_token(auth_data, expected_status=http.client.CREATED)

    def test_with_passcode_in_previous_windows_default(self):
        """Confirm previous window default of 1 works."""
        creds = self._make_credentials('totp')
        secret = creds[-1]['blob']
        past = datetime.datetime.utcnow() - datetime.timedelta(seconds=30)
        with freezegun.freeze_time(past):
            auth_data = self._make_auth_data_by_id(totp._generate_totp_passcodes(secret)[0])
        self.useFixture(fixture.TimeFixture())
        self.v3_create_token(auth_data, expected_status=http.client.CREATED)

    def test_with_passcode_in_previous_windows_extended(self):
        self.config_fixture.config(group='totp', included_previous_windows=4)
        creds = self._make_credentials('totp')
        secret = creds[-1]['blob']
        past = datetime.datetime.utcnow() - datetime.timedelta(minutes=2)
        self.useFixture(fixture.TimeFixture(past))
        auth_data = self._make_auth_data_by_id(totp._generate_totp_passcodes(secret)[0])
        self.useFixture(fixture.TimeFixture())
        self.v3_create_token(auth_data, expected_status=http.client.CREATED)

    def test_with_an_invalid_passcode_and_user_credentials(self):
        self._make_credentials('totp')
        auth_data = self._make_auth_data_by_id('000000')
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_with_an_invalid_passcode_with_no_user_credentials(self):
        auth_data = self._make_auth_data_by_id('000000')
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_with_a_corrupt_totp_credential(self):
        self._make_credentials('totp', count=1, blob='0')
        auth_data = self._make_auth_data_by_id('000000')
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_with_multiple_credentials(self):
        self._make_credentials('other', 3)
        creds = self._make_credentials('totp', count=3)
        secret = creds[-1]['blob']
        self.useFixture(fixture.TimeFixture())
        auth_data = self._make_auth_data_by_id(totp._generate_totp_passcodes(secret)[0])
        self.v3_create_token(auth_data, expected_status=http.client.CREATED)

    def test_with_multiple_users(self):
        self._make_credentials('totp', count=3)
        user = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
        PROVIDERS.assignment_api.create_grant(self.role['id'], user_id=user['id'], project_id=self.project['id'])
        creds = self._make_credentials('totp', count=1, user_id=user['id'])
        secret = creds[-1]['blob']
        self.useFixture(fixture.TimeFixture())
        auth_data = self._make_auth_data_by_id(totp._generate_totp_passcodes(secret)[0], user_id=user['id'])
        self.v3_create_token(auth_data, expected_status=http.client.CREATED)

    def test_with_multiple_users_and_invalid_credentials(self):
        """Prevent logging in with someone else's credentials.

        It's very easy to forget to limit the credentials query by user.
        Let's just test it for a sanity check.
        """
        self._make_credentials('totp', count=3)
        new_user = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
        PROVIDERS.assignment_api.create_grant(self.role['id'], user_id=new_user['id'], project_id=self.project['id'])
        user2_creds = self._make_credentials('totp', count=1, user_id=new_user['id'])
        user_id = self.default_domain_user['id']
        secret = user2_creds[-1]['blob']
        auth_data = self._make_auth_data_by_id(totp._generate_totp_passcodes(secret)[0], user_id=user_id)
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_with_username_and_domain_id(self):
        creds = self._make_credentials('totp')
        secret = creds[-1]['blob']
        self.useFixture(fixture.TimeFixture())
        auth_data = self._make_auth_data_by_name(totp._generate_totp_passcodes(secret)[0], username=self.default_domain_user['name'], user_domain_id=self.default_domain_user['domain_id'])
        self.v3_create_token(auth_data, expected_status=http.client.CREATED)

    def test_generated_passcode_is_correct_format(self):
        secret = self._make_credentials('totp')[-1]['blob']
        passcode = totp._generate_totp_passcodes(secret)[0]
        reg = re.compile('^-?[0-9]+$')
        self.assertTrue(reg.match(passcode))