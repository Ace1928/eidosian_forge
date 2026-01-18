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
class TestTrustChain(test_v3.RestfulTestCase):

    def config_overrides(self):
        super(TestTrustChain, self).config_overrides()
        self.config_fixture.config(group='trust', allow_redelegation=True, max_redelegation_count=10)

    def setUp(self):
        super(TestTrustChain, self).setUp()
        'Create a trust chain using redelegation.\n\n        A trust chain is a series of trusts that are redelegated. For example,\n        self.user_list consists of userA, userB, and userC. The first trust in\n        the trust chain is going to be established between self.user and userA,\n        call it trustA. Then, userA is going to obtain a trust scoped token\n        using trustA, and with that token create a trust between userA and\n        userB called trustB. This pattern will continue with userB creating a\n        trust with userC.\n        So the trust chain should look something like:\n            trustA -> trustB -> trustC\n        Where:\n            self.user is trusting userA with trustA\n            userA is trusting userB with trustB\n            userB is trusting userC with trustC\n\n        '
        self.user_list = list()
        self.trust_chain = list()
        for _ in range(3):
            user = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
            self.user_list.append(user)
        trustee = self.user_list[0]
        trust_ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=trustee['id'], project_id=self.project_id, impersonation=True, expires=dict(minutes=1), role_ids=[self.role_id], allow_redelegation=True, redelegation_count=3)
        r = self.post('/OS-TRUST/trusts', body={'trust': trust_ref})
        trust = self.assertValidTrustResponse(r)
        auth_data = self.build_authentication_request(user_id=trustee['id'], password=trustee['password'], trust_id=trust['id'])
        trust_token = self.get_requested_token(auth_data)
        self.trust_chain.append(trust)
        for next_trustee in self.user_list[1:]:
            trust_ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=next_trustee['id'], project_id=self.project_id, impersonation=True, role_ids=[self.role_id], allow_redelegation=True)
            r = self.post('/OS-TRUST/trusts', body={'trust': trust_ref}, token=trust_token)
            trust = self.assertValidTrustResponse(r)
            auth_data = self.build_authentication_request(user_id=next_trustee['id'], password=next_trustee['password'], trust_id=trust['id'])
            trust_token = self.get_requested_token(auth_data)
            self.trust_chain.append(trust)
        trustee = self.user_list[-1]
        trust = self.trust_chain[-1]
        auth_data = self.build_authentication_request(user_id=trustee['id'], password=trustee['password'], trust_id=trust['id'])
        self.last_token = self.get_requested_token(auth_data)

    def assert_user_authenticate(self, user):
        auth_data = self.build_authentication_request(user_id=user['id'], password=user['password'])
        r = self.v3_create_token(auth_data)
        self.assertValidTokenResponse(r)

    def assert_trust_tokens_revoked(self, trust_id):
        trustee = self.user_list[0]
        auth_data = self.build_authentication_request(user_id=trustee['id'], password=trustee['password'])
        r = self.v3_create_token(auth_data)
        self.assertValidTokenResponse(r)
        revocation_response = self.get('/OS-REVOKE/events')
        revocation_events = revocation_response.json_body['events']
        found = False
        for event in revocation_events:
            if event.get('OS-TRUST:trust_id') == trust_id:
                found = True
        self.assertTrue(found, 'event with trust_id %s not found in list' % trust_id)

    def test_delete_trust_cascade(self):
        self.assert_user_authenticate(self.user_list[0])
        self.delete('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': self.trust_chain[0]['id']})
        headers = {'X-Subject-Token': self.last_token}
        self.head('/auth/tokens', headers=headers, expected_status=http.client.NOT_FOUND)
        self.assert_trust_tokens_revoked(self.trust_chain[0]['id'])

    def test_delete_broken_chain(self):
        self.assert_user_authenticate(self.user_list[0])
        self.delete('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': self.trust_chain[0]['id']})
        for i in range(len(self.user_list) - 1):
            auth_data = self.build_authentication_request(user_id=self.user_list[i]['id'], password=self.user_list[i]['password'])
            auth_token = self.get_requested_token(auth_data)
            self.get('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': self.trust_chain[i + 1]['id']}, token=auth_token, expected_status=http.client.NOT_FOUND)

    def test_trustor_roles_revoked(self):
        self.assert_user_authenticate(self.user_list[0])
        PROVIDERS.assignment_api.remove_role_from_user_and_project(self.user_id, self.project_id, self.role_id)
        for i in range(len(self.user_list[1:])):
            trustee = self.user_list[i]
            auth_data = self.build_authentication_request(user_id=trustee['id'], password=trustee['password'])
            token = self.get_requested_token(auth_data)
            auth_data = self.build_authentication_request(token=token, trust_id=self.trust_chain[i - 1]['id'])
            self.v3_create_token(auth_data, expected_status=http.client.FORBIDDEN)

    def test_intermediate_user_disabled(self):
        self.assert_user_authenticate(self.user_list[0])
        disabled = self.user_list[0]
        disabled['enabled'] = False
        PROVIDERS.identity_api.update_user(disabled['id'], disabled)
        with mock.patch.object(policy, 'enforce', return_value=True):
            headers = {'X-Subject-Token': self.last_token}
            self.head('/auth/tokens', headers=headers, expected_status=http.client.FORBIDDEN)

    def test_intermediate_user_deleted(self):
        self.assert_user_authenticate(self.user_list[0])
        PROVIDERS.identity_api.delete_user(self.user_list[0]['id'])
        with mock.patch.object(policy, 'enforce', return_value=True):
            headers = {'X-Subject-Token': self.last_token}
            self.head('/auth/tokens', headers=headers, expected_status=http.client.NOT_FOUND)