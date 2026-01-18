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
class TestFernetTokenAPIs(test_v3.RestfulTestCase, TokenAPITests, TokenDataTests):

    def config_overrides(self):
        super(TestFernetTokenAPIs, self).config_overrides()
        self.config_fixture.config(group='token', provider='fernet', cache_on_issue=True)
        self.useFixture(ksfixtures.KeyRepository(self.config_fixture, 'fernet_tokens', CONF.fernet_tokens.max_active_keys))

    def setUp(self):
        super(TestFernetTokenAPIs, self).setUp()
        self.doSetUp()

    def _make_auth_request(self, auth_data):
        token = super(TestFernetTokenAPIs, self)._make_auth_request(auth_data)
        self.assertLess(len(token), 255)
        return token

    def test_validate_tampered_unscoped_token_fails(self):
        unscoped_token = self._get_unscoped_token()
        tampered_token = unscoped_token[:50] + uuid.uuid4().hex + unscoped_token[50 + 32:]
        self._validate_token(tampered_token, expected_status=http.client.NOT_FOUND)

    def test_validate_tampered_project_scoped_token_fails(self):
        project_scoped_token = self._get_project_scoped_token()
        tampered_token = project_scoped_token[:50] + uuid.uuid4().hex + project_scoped_token[50 + 32:]
        self._validate_token(tampered_token, expected_status=http.client.NOT_FOUND)

    def test_validate_tampered_trust_scoped_token_fails(self):
        trustee_user, trust = self._create_trust()
        trust_scoped_token = self._get_trust_scoped_token(trustee_user, trust)
        tampered_token = trust_scoped_token[:50] + uuid.uuid4().hex + trust_scoped_token[50 + 32:]
        self._validate_token(tampered_token, expected_status=http.client.NOT_FOUND)

    def test_trust_scoped_token_is_invalid_after_disabling_trustor(self):
        trustee_user, trust = self._create_trust()
        trust_scoped_token = self._get_trust_scoped_token(trustee_user, trust)
        r = self._validate_token(trust_scoped_token)
        self.assertValidProjectScopedTokenResponse(r)
        trustor_update_ref = dict(enabled=False)
        PROVIDERS.identity_api.update_user(self.user['id'], trustor_update_ref)
        self._validate_token(trust_scoped_token, expected_status=http.client.FORBIDDEN)