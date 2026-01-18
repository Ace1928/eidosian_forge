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
class CredentialTestCase(CredentialBaseTestCase):
    """Test credential CRUD."""

    def setUp(self):
        super(CredentialTestCase, self).setUp()
        self.credential = unit.new_credential_ref(user_id=self.user['id'], project_id=self.project_id)
        PROVIDERS.credential_api.create_credential(self.credential['id'], self.credential)

    def test_credential_api_delete_credentials_for_project(self):
        PROVIDERS.credential_api.delete_credentials_for_project(self.project_id)
        self.assertRaises(exception.CredentialNotFound, PROVIDERS.credential_api.get_credential, credential_id=self.credential['id'])

    def test_credential_api_delete_credentials_for_user(self):
        PROVIDERS.credential_api.delete_credentials_for_user(self.user_id)
        self.assertRaises(exception.CredentialNotFound, PROVIDERS.credential_api.get_credential, credential_id=self.credential['id'])

    def test_list_credentials(self):
        """Call ``GET /credentials``."""
        r = self.get('/credentials')
        self.assertValidCredentialListResponse(r, ref=self.credential)

    def test_list_credentials_filtered_by_user_id(self):
        """Call ``GET  /credentials?user_id={user_id}``."""
        credential = unit.new_credential_ref(user_id=uuid.uuid4().hex)
        PROVIDERS.credential_api.create_credential(credential['id'], credential)
        r = self.get('/credentials?user_id=%s' % self.user['id'])
        self.assertValidCredentialListResponse(r, ref=self.credential)
        for cred in r.result['credentials']:
            self.assertEqual(self.user['id'], cred['user_id'])

    def test_list_credentials_filtered_by_type(self):
        """Call ``GET  /credentials?type={type}``."""
        PROVIDERS.assignment_api.create_system_grant_for_user(self.user_id, self.role_id)
        token = self.get_system_scoped_token()
        ec2_credential = unit.new_credential_ref(user_id=uuid.uuid4().hex, project_id=self.project_id, type=CRED_TYPE_EC2)
        ec2_resp = PROVIDERS.credential_api.create_credential(ec2_credential['id'], ec2_credential)
        r = self.get('/credentials?type=cert', token=token)
        self.assertValidCredentialListResponse(r, ref=self.credential)
        for cred in r.result['credentials']:
            self.assertEqual('cert', cred['type'])
        r_ec2 = self.get('/credentials?type=ec2', token=token)
        self.assertThat(r_ec2.result['credentials'], matchers.HasLength(1))
        cred_ec2 = r_ec2.result['credentials'][0]
        self.assertValidCredentialListResponse(r_ec2, ref=ec2_resp)
        self.assertEqual(CRED_TYPE_EC2, cred_ec2['type'])
        self.assertEqual(ec2_credential['id'], cred_ec2['id'])

    def test_list_credentials_filtered_by_type_and_user_id(self):
        """Call ``GET  /credentials?user_id={user_id}&type={type}``."""
        user1_id = uuid.uuid4().hex
        user2_id = uuid.uuid4().hex
        PROVIDERS.assignment_api.create_system_grant_for_user(self.user_id, self.role_id)
        token = self.get_system_scoped_token()
        credential_user1_ec2 = unit.new_credential_ref(user_id=user1_id, type=CRED_TYPE_EC2)
        credential_user1_cert = unit.new_credential_ref(user_id=user1_id)
        credential_user2_cert = unit.new_credential_ref(user_id=user2_id)
        PROVIDERS.credential_api.create_credential(credential_user1_ec2['id'], credential_user1_ec2)
        PROVIDERS.credential_api.create_credential(credential_user1_cert['id'], credential_user1_cert)
        PROVIDERS.credential_api.create_credential(credential_user2_cert['id'], credential_user2_cert)
        r = self.get('/credentials?user_id=%s&type=ec2' % user1_id, token=token)
        self.assertValidCredentialListResponse(r, ref=credential_user1_ec2)
        self.assertThat(r.result['credentials'], matchers.HasLength(1))
        cred = r.result['credentials'][0]
        self.assertEqual(CRED_TYPE_EC2, cred['type'])
        self.assertEqual(user1_id, cred['user_id'])

    def test_create_credential(self):
        """Call ``POST /credentials``."""
        ref = unit.new_credential_ref(user_id=self.user['id'])
        r = self.post('/credentials', body={'credential': ref})
        self.assertValidCredentialResponse(r, ref)

    def test_get_credential(self):
        """Call ``GET /credentials/{credential_id}``."""
        r = self.get('/credentials/%(credential_id)s' % {'credential_id': self.credential['id']})
        self.assertValidCredentialResponse(r, self.credential)

    def test_update_credential(self):
        """Call ``PATCH /credentials/{credential_id}``."""
        ref = unit.new_credential_ref(user_id=self.user['id'], project_id=self.project_id)
        del ref['id']
        r = self.patch('/credentials/%(credential_id)s' % {'credential_id': self.credential['id']}, body={'credential': ref})
        self.assertValidCredentialResponse(r, ref)

    def test_update_credential_to_ec2_type(self):
        """Call ``PATCH /credentials/{credential_id}``."""
        ref = unit.new_credential_ref(user_id=self.user['id'])
        r = self.post('/credentials', body={'credential': ref})
        self.assertValidCredentialResponse(r, ref)
        credential_id = r.result.get('credential')['id']
        update_ref = {'type': 'ec2', 'project_id': self.project_id}
        self.patch('/credentials/%(credential_id)s' % {'credential_id': credential_id}, body={'credential': update_ref})

    def test_update_credential_to_ec2_missing_project_id(self):
        """Call ``PATCH /credentials/{credential_id}``."""
        ref = unit.new_credential_ref(user_id=self.user['id'])
        r = self.post('/credentials', body={'credential': ref})
        self.assertValidCredentialResponse(r, ref)
        credential_id = r.result.get('credential')['id']
        update_ref = {'type': 'ec2'}
        self.patch('/credentials/%(credential_id)s' % {'credential_id': credential_id}, body={'credential': update_ref}, expected_status=http.client.BAD_REQUEST)

    def test_update_credential_to_ec2_with_previously_set_project_id(self):
        """Call ``PATCH /credentials/{credential_id}``."""
        ref = unit.new_credential_ref(user_id=self.user['id'], project_id=self.project_id)
        r = self.post('/credentials', body={'credential': ref})
        self.assertValidCredentialResponse(r, ref)
        credential_id = r.result.get('credential')['id']
        update_ref = {'type': 'ec2'}
        self.patch('/credentials/%(credential_id)s' % {'credential_id': credential_id}, body={'credential': update_ref})

    def test_update_credential_non_owner(self):
        """Call ``PATCH /credentials/{credential_id}``."""
        alt_user = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
        alt_user_id = alt_user['id']
        alt_project = unit.new_project_ref(domain_id=self.domain_id)
        alt_project_id = alt_project['id']
        PROVIDERS.resource_api.create_project(alt_project['id'], alt_project)
        alt_role = unit.new_role_ref(name='reader')
        alt_role_id = alt_role['id']
        PROVIDERS.role_api.create_role(alt_role_id, alt_role)
        PROVIDERS.assignment_api.add_role_to_user_and_project(alt_user_id, alt_project_id, alt_role_id)
        auth = self.build_authentication_request(user_id=alt_user_id, password=alt_user['password'], project_id=alt_project_id)
        ref = unit.new_credential_ref(user_id=alt_user_id, project_id=alt_project_id)
        r = self.post('/credentials', auth=auth, body={'credential': ref})
        self.assertValidCredentialResponse(r, ref)
        credential_id = r.result.get('credential')['id']
        update_ref = {'user_id': self.user_id, 'project_id': self.project_id}
        self.patch('/credentials/%(credential_id)s' % {'credential_id': credential_id}, expected_status=403, auth=auth, body={'credential': update_ref})

    def test_update_ec2_credential_change_trust_id(self):
        """Call ``PATCH /credentials/{credential_id}``."""
        blob, ref = unit.new_ec2_credential(user_id=self.user['id'], project_id=self.project_id)
        blob['trust_id'] = uuid.uuid4().hex
        ref['blob'] = json.dumps(blob)
        r = self.post('/credentials', body={'credential': ref})
        self.assertValidCredentialResponse(r, ref)
        credential_id = r.result.get('credential')['id']
        blob['trust_id'] = uuid.uuid4().hex
        update_ref = {'blob': json.dumps(blob)}
        self.patch('/credentials/%(credential_id)s' % {'credential_id': credential_id}, body={'credential': update_ref}, expected_status=http.client.BAD_REQUEST)
        del blob['trust_id']
        update_ref = {'blob': json.dumps(blob)}
        self.patch('/credentials/%(credential_id)s' % {'credential_id': credential_id}, body={'credential': update_ref}, expected_status=http.client.BAD_REQUEST)

    def test_update_ec2_credential_change_app_cred_id(self):
        """Call ``PATCH /credentials/{credential_id}``."""
        blob, ref = unit.new_ec2_credential(user_id=self.user['id'], project_id=self.project_id)
        blob['app_cred_id'] = uuid.uuid4().hex
        ref['blob'] = json.dumps(blob)
        r = self.post('/credentials', body={'credential': ref})
        self.assertValidCredentialResponse(r, ref)
        credential_id = r.result.get('credential')['id']
        blob['app_cred_id'] = uuid.uuid4().hex
        update_ref = {'blob': json.dumps(blob)}
        self.patch('/credentials/%(credential_id)s' % {'credential_id': credential_id}, body={'credential': update_ref}, expected_status=http.client.BAD_REQUEST)
        del blob['app_cred_id']
        update_ref = {'blob': json.dumps(blob)}
        self.patch('/credentials/%(credential_id)s' % {'credential_id': credential_id}, body={'credential': update_ref}, expected_status=http.client.BAD_REQUEST)

    def test_update_ec2_credential_change_access_token_id(self):
        """Call ``PATCH /credentials/{credential_id}``."""
        blob, ref = unit.new_ec2_credential(user_id=self.user['id'], project_id=self.project_id)
        blob['access_token_id'] = uuid.uuid4().hex
        ref['blob'] = json.dumps(blob)
        r = self.post('/credentials', body={'credential': ref})
        self.assertValidCredentialResponse(r, ref)
        credential_id = r.result.get('credential')['id']
        blob['access_token_id'] = uuid.uuid4().hex
        update_ref = {'blob': json.dumps(blob)}
        self.patch('/credentials/%(credential_id)s' % {'credential_id': credential_id}, body={'credential': update_ref}, expected_status=http.client.BAD_REQUEST)
        del blob['access_token_id']
        update_ref = {'blob': json.dumps(blob)}
        self.patch('/credentials/%(credential_id)s' % {'credential_id': credential_id}, body={'credential': update_ref}, expected_status=http.client.BAD_REQUEST)

    def test_update_ec2_credential_change_access_id(self):
        """Call ``PATCH /credentials/{credential_id}``."""
        blob, ref = unit.new_ec2_credential(user_id=self.user['id'], project_id=self.project_id)
        blob['access_id'] = uuid.uuid4().hex
        ref['blob'] = json.dumps(blob)
        r = self.post('/credentials', body={'credential': ref})
        self.assertValidCredentialResponse(r, ref)
        credential_id = r.result.get('credential')['id']
        blob['access_id'] = uuid.uuid4().hex
        update_ref = {'blob': json.dumps(blob)}
        self.patch('/credentials/%(credential_id)s' % {'credential_id': credential_id}, body={'credential': update_ref}, expected_status=http.client.BAD_REQUEST)
        del blob['access_id']
        update_ref = {'blob': json.dumps(blob)}
        self.patch('/credentials/%(credential_id)s' % {'credential_id': credential_id}, body={'credential': update_ref}, expected_status=http.client.BAD_REQUEST)

    def test_delete_credential(self):
        """Call ``DELETE /credentials/{credential_id}``."""
        self.delete('/credentials/%(credential_id)s' % {'credential_id': self.credential['id']})

    def test_delete_credential_retries_on_deadlock(self):
        patcher = mock.patch('sqlalchemy.orm.query.Query.delete', autospec=True)

        class FakeDeadlock(object):

            def __init__(self, mock_patcher):
                self.deadlock_count = 2
                self.mock_patcher = mock_patcher
                self.patched = True

            def __call__(self, *args, **kwargs):
                if self.deadlock_count > 1:
                    self.deadlock_count -= 1
                else:
                    self.mock_patcher.stop()
                    self.patched = False
                raise oslo_db_exception.DBDeadlock
        sql_delete_mock = patcher.start()
        side_effect = FakeDeadlock(patcher)
        sql_delete_mock.side_effect = side_effect
        try:
            PROVIDERS.credential_api.delete_credentials_for_user(user_id=self.user['id'])
        finally:
            if side_effect.patched:
                patcher.stop()
        self.assertEqual(sql_delete_mock.call_count, 2)

    def test_create_ec2_credential(self):
        """Call ``POST /credentials`` for creating ec2 credential."""
        blob, ref = unit.new_ec2_credential(user_id=self.user['id'], project_id=self.project_id)
        r = self.post('/credentials', body={'credential': ref})
        self.assertValidCredentialResponse(r, ref)
        access = blob['access'].encode('utf-8')
        self.assertEqual(hashlib.sha256(access).hexdigest(), r.result['credential']['id'])
        self.post('/credentials', body={'credential': ref}, expected_status=http.client.CONFLICT)

    def test_get_ec2_dict_blob(self):
        """Ensure non-JSON blob data is correctly converted."""
        expected_blob, credential_id = self._create_dict_blob_credential()
        r = self.get('/credentials/%(credential_id)s' % {'credential_id': credential_id})
        self.assertEqual(json.loads(expected_blob), json.loads(r.result['credential']['blob']))

    def test_list_ec2_dict_blob(self):
        """Ensure non-JSON blob data is correctly converted."""
        expected_blob, credential_id = self._create_dict_blob_credential()
        list_r = self.get('/credentials')
        list_creds = list_r.result['credentials']
        list_ids = [r['id'] for r in list_creds]
        self.assertIn(credential_id, list_ids)
        for r in list_creds:
            if r['id'] == credential_id:
                self.assertEqual(json.loads(expected_blob), json.loads(r['blob']))

    def test_create_non_ec2_credential(self):
        """Test creating non-ec2 credential.

        Call ``POST /credentials``.
        """
        blob, ref = unit.new_cert_credential(user_id=self.user['id'])
        r = self.post('/credentials', body={'credential': ref})
        self.assertValidCredentialResponse(r, ref)
        access = blob['access'].encode('utf-8')
        self.assertNotEqual(hashlib.sha256(access).hexdigest(), r.result['credential']['id'])

    def test_create_ec2_credential_with_missing_project_id(self):
        """Test Creating ec2 credential with missing project_id.

        Call ``POST /credentials``.
        """
        _, ref = unit.new_ec2_credential(user_id=self.user['id'], project_id=None)
        self.post('/credentials', body={'credential': ref}, expected_status=http.client.BAD_REQUEST)

    def test_create_ec2_credential_with_invalid_blob(self):
        """Test creating ec2 credential with invalid blob.

        Call ``POST /credentials``.
        """
        ref = unit.new_credential_ref(user_id=self.user['id'], project_id=self.project_id, blob='{"abc":"def"d}', type=CRED_TYPE_EC2)
        response = self.post('/credentials', body={'credential': ref}, expected_status=http.client.BAD_REQUEST)
        self.assertValidErrorResponse(response)

    def test_create_credential_with_admin_token(self):
        ref = unit.new_credential_ref(user_id=self.user['id'])
        r = self.post('/credentials', body={'credential': ref}, token=self.get_admin_token())
        self.assertValidCredentialResponse(r, ref)