import hashlib
import json
from unittest import mock
import uuid
import http.client
from keystoneclient.contrib.ec2 import utils as ec2_utils
from oslo_db import exception as oslo_db_exception
from testtools import matchers
import urllib
from keystone.api import ec2tokens
from keystone.common import provider_api
from keystone.common import utils
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone import oauth1
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
class TestCredentialTrustScoped(CredentialBaseTestCase):
    """Test credential with trust scoped token."""

    def setUp(self):
        super(TestCredentialTrustScoped, self).setUp()
        self.trustee_user = unit.new_user_ref(domain_id=self.domain_id)
        password = self.trustee_user['password']
        self.trustee_user = PROVIDERS.identity_api.create_user(self.trustee_user)
        self.trustee_user['password'] = password
        self.trustee_user_id = self.trustee_user['id']
        self.useFixture(ksfixtures.KeyRepository(self.config_fixture, 'credential', credential_fernet.MAX_ACTIVE_KEYS))

    def config_overrides(self):
        super(TestCredentialTrustScoped, self).config_overrides()
        self.config_fixture.config(group='trust')

    def test_trust_scoped_ec2_credential(self):
        """Test creating trust scoped ec2 credential.

        Call ``POST /credentials``.
        """
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, impersonation=True, expires=dict(minutes=1), role_ids=[self.role_id])
        del ref['id']
        r = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = self.assertValidTrustResponse(r)
        auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'], trust_id=trust['id'])
        r = self.v3_create_token(auth_data)
        self.assertValidProjectScopedTokenResponse(r, self.user)
        trust_id = r.result['token']['OS-TRUST:trust']['id']
        token_id = r.headers.get('X-Subject-Token')
        blob, ref = unit.new_ec2_credential(user_id=self.user_id, project_id=self.project_id)
        r = self.post('/credentials', body={'credential': ref}, token=token_id)
        ret_ref = ref.copy()
        ret_blob = blob.copy()
        ret_blob['trust_id'] = trust_id
        ret_ref['blob'] = json.dumps(ret_blob)
        self.assertValidCredentialResponse(r, ref=ret_ref)
        access = blob['access'].encode('utf-8')
        self.assertEqual(hashlib.sha256(access).hexdigest(), r.result['credential']['id'])
        role = unit.new_role_ref(name='reader')
        role_id = role['id']
        PROVIDERS.role_api.create_role(role_id, role)
        PROVIDERS.assignment_api.add_role_to_user_and_project(self.user_id, self.project_id, role_id)
        ret_blob = json.loads(r.result['credential']['blob'])
        ec2token = self._test_get_token(access=ret_blob['access'], secret=ret_blob['secret'])
        ec2_roles = [role['id'] for role in ec2token['roles']]
        self.assertIn(self.role_id, ec2_roles)
        self.assertNotIn(role_id, ec2_roles)
        self.post('/credentials', body={'credential': ref}, token=token_id, expected_status=http.client.CONFLICT)