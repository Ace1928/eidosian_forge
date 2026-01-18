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
class TestCredentialEc2(CredentialBaseTestCase):
    """Test v3 credential compatibility with ec2tokens."""

    def test_ec2_credential_signature_validate(self):
        """Test signature validation with a v3 ec2 credential."""
        blob, ref = unit.new_ec2_credential(user_id=self.user['id'], project_id=self.project_id)
        r = self.post('/credentials', body={'credential': ref})
        self.assertValidCredentialResponse(r, ref)
        access = blob['access'].encode('utf-8')
        self.assertEqual(hashlib.sha256(access).hexdigest(), r.result['credential']['id'])
        cred_blob = json.loads(r.result['credential']['blob'])
        self.assertEqual(blob, cred_blob)
        self._test_get_token(access=cred_blob['access'], secret=cred_blob['secret'])

    def test_ec2_credential_signature_validate_legacy(self):
        """Test signature validation with a legacy v3 ec2 credential."""
        cred_json, _ = self._create_dict_blob_credential()
        cred_blob = json.loads(cred_json)
        self._test_get_token(access=cred_blob['access'], secret=cred_blob['secret'])

    def _get_ec2_cred_uri(self):
        return '/users/%s/credentials/OS-EC2' % self.user_id

    def _get_ec2_cred(self):
        uri = self._get_ec2_cred_uri()
        r = self.post(uri, body={'tenant_id': self.project_id})
        return r.result['credential']

    def test_ec2_create_credential(self):
        """Test ec2 credential creation."""
        ec2_cred = self._get_ec2_cred()
        self.assertEqual(self.user_id, ec2_cred['user_id'])
        self.assertEqual(self.project_id, ec2_cred['tenant_id'])
        self.assertIsNone(ec2_cred['trust_id'])
        self._test_get_token(access=ec2_cred['access'], secret=ec2_cred['secret'])
        uri = '/'.join([self._get_ec2_cred_uri(), ec2_cred['access']])
        self.assertThat(ec2_cred['links']['self'], matchers.EndsWith(uri))

    def test_ec2_get_credential(self):
        ec2_cred = self._get_ec2_cred()
        uri = '/'.join([self._get_ec2_cred_uri(), ec2_cred['access']])
        r = self.get(uri)
        self.assertDictEqual(ec2_cred, r.result['credential'])
        self.assertThat(ec2_cred['links']['self'], matchers.EndsWith(uri))

    def test_ec2_cannot_get_non_ec2_credential(self):
        access_key = uuid.uuid4().hex
        cred_id = utils.hash_access_key(access_key)
        non_ec2_cred = unit.new_credential_ref(user_id=self.user_id, project_id=self.project_id)
        non_ec2_cred['id'] = cred_id
        PROVIDERS.credential_api.create_credential(cred_id, non_ec2_cred)
        uri = '/'.join([self._get_ec2_cred_uri(), access_key])
        self.get(uri, expected_status=http.client.UNAUTHORIZED)

    def test_ec2_list_credentials(self):
        """Test ec2 credential listing."""
        self._get_ec2_cred()
        uri = self._get_ec2_cred_uri()
        r = self.get(uri)
        cred_list = r.result['credentials']
        self.assertEqual(1, len(cred_list))
        self.assertThat(r.result['links']['self'], matchers.EndsWith(uri))
        non_ec2_cred = unit.new_credential_ref(user_id=self.user_id, project_id=self.project_id)
        non_ec2_cred['type'] = uuid.uuid4().hex
        PROVIDERS.credential_api.create_credential(non_ec2_cred['id'], non_ec2_cred)
        r = self.get(uri)
        cred_list_2 = r.result['credentials']
        self.assertEqual(1, len(cred_list_2))
        self.assertEqual(cred_list[0], cred_list_2[0])

    def test_ec2_delete_credential(self):
        """Test ec2 credential deletion."""
        ec2_cred = self._get_ec2_cred()
        uri = '/'.join([self._get_ec2_cred_uri(), ec2_cred['access']])
        cred_from_credential_api = PROVIDERS.credential_api.list_credentials_for_user(self.user_id, type=CRED_TYPE_EC2)
        self.assertEqual(1, len(cred_from_credential_api))
        self.delete(uri)
        self.assertRaises(exception.CredentialNotFound, PROVIDERS.credential_api.get_credential, cred_from_credential_api[0]['id'])