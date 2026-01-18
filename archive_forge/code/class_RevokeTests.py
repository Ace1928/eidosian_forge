import datetime
from unittest import mock
import uuid
from oslo_utils import timeutils
from testtools import matchers
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.models import revoke_model
from keystone.revoke.backends import sql
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_backend_sql
from keystone.token import provider
class RevokeTests(object):

    def _assertTokenRevoked(self, token_data):
        self.assertRaises(exception.TokenNotFound, PROVIDERS.revoke_api.check_token, token=token_data)

    def _assertTokenNotRevoked(self, token_data):
        self.assertIsNone(PROVIDERS.revoke_api.check_token(token_data))

    def test_list(self):
        PROVIDERS.revoke_api.revoke_by_user(user_id=1)
        self.assertEqual(1, len(PROVIDERS.revoke_api.list_events()))
        PROVIDERS.revoke_api.revoke_by_user(user_id=2)
        self.assertEqual(2, len(PROVIDERS.revoke_api.list_events()))

    def test_list_since(self):
        PROVIDERS.revoke_api.revoke_by_user(user_id=1)
        PROVIDERS.revoke_api.revoke_by_user(user_id=2)
        past = timeutils.utcnow() - datetime.timedelta(seconds=1000)
        self.assertEqual(2, len(PROVIDERS.revoke_api.list_events(last_fetch=past)))
        future = timeutils.utcnow() + datetime.timedelta(seconds=1000)
        self.assertEqual(0, len(PROVIDERS.revoke_api.list_events(last_fetch=future)))

    def test_list_revoked_user(self):
        revocation_backend = sql.Revoke()
        first_token = _sample_blank_token()
        first_token['user_id'] = uuid.uuid4().hex
        PROVIDERS.revoke_api.revoke_by_user(user_id=first_token['user_id'])
        self._assertTokenRevoked(first_token)
        self.assertEqual(1, len(revocation_backend.list_events(token=first_token)))
        second_token = _sample_blank_token()
        second_token['user_id'] = uuid.uuid4().hex
        PROVIDERS.revoke_api.revoke_by_user(user_id=second_token['user_id'])
        self._assertTokenRevoked(second_token)
        self.assertEqual(1, len(revocation_backend.list_events(token=second_token)))
        third_token = _sample_blank_token()
        third_token['user_id'] = uuid.uuid4().hex
        self._assertTokenNotRevoked(third_token)
        self.assertEqual(0, len(revocation_backend.list_events(token=third_token)))
        fourth_token = _sample_blank_token()
        fourth_token['user_id'] = None
        self._assertTokenNotRevoked(fourth_token)
        self.assertEqual(0, len(revocation_backend.list_events(token=fourth_token)))

    def test_list_revoked_project(self):
        revocation_backend = sql.Revoke()
        token = _sample_blank_token()
        first_token = _sample_blank_token()
        first_token['project_id'] = uuid.uuid4().hex
        revocation_backend.revoke(revoke_model.RevokeEvent(project_id=first_token['project_id']))
        self._assertTokenRevoked(first_token)
        self.assertEqual(1, len(revocation_backend.list_events(token=first_token)))
        second_token = _sample_blank_token()
        second_token['project_id'] = uuid.uuid4().hex
        revocation_backend.revoke(revoke_model.RevokeEvent(project_id=second_token['project_id']))
        self._assertTokenRevoked(second_token)
        self.assertEqual(1, len(revocation_backend.list_events(token=second_token)))
        third_token = _sample_blank_token()
        third_token['project_id'] = None
        self._assertTokenNotRevoked(token)
        self.assertEqual(0, len(revocation_backend.list_events(token=token)))

    def test_list_revoked_audit(self):
        revocation_backend = sql.Revoke()
        first_token = _sample_blank_token()
        first_token['audit_id'] = provider.random_urlsafe_str()
        PROVIDERS.revoke_api.revoke_by_audit_id(audit_id=first_token['audit_id'])
        self._assertTokenRevoked(first_token)
        self.assertEqual(1, len(revocation_backend.list_events(token=first_token)))
        second_token = _sample_blank_token()
        second_token['audit_id'] = provider.random_urlsafe_str()
        PROVIDERS.revoke_api.revoke_by_audit_id(audit_id=second_token['audit_id'])
        self._assertTokenRevoked(second_token)
        self.assertEqual(1, len(revocation_backend.list_events(token=second_token)))
        third_token = _sample_blank_token()
        third_token['audit_id'] = None
        self._assertTokenNotRevoked(third_token)
        self.assertEqual(0, len(revocation_backend.list_events(token=third_token)))

    def test_list_revoked_since(self):
        revocation_backend = sql.Revoke()
        token = _sample_blank_token()
        PROVIDERS.revoke_api.revoke_by_user(user_id=None)
        PROVIDERS.revoke_api.revoke_by_user(user_id=None)
        self.assertEqual(2, len(revocation_backend.list_events(token=token)))
        future = timeutils.utcnow() + datetime.timedelta(seconds=1000)
        token['issued_at'] = future
        self.assertEqual(0, len(revocation_backend.list_events(token=token)))

    def test_list_revoked_multiple_filters(self):
        revocation_backend = sql.Revoke()
        first_token = _sample_blank_token()
        first_token['user_id'] = uuid.uuid4().hex
        first_token['project_id'] = uuid.uuid4().hex
        first_token['audit_id'] = provider.random_urlsafe_str()
        PROVIDERS.revoke_api.revoke(revoke_model.RevokeEvent(user_id=first_token['user_id'], project_id=first_token['project_id'], audit_id=first_token['audit_id']))
        self._assertTokenRevoked(first_token)
        self.assertEqual(1, len(revocation_backend.list_events(token=first_token)))
        second_token = _sample_blank_token()
        self._assertTokenNotRevoked(second_token)
        self.assertEqual(0, len(revocation_backend.list_events(token=second_token)))
        third_token = _sample_blank_token()
        third_token['project_id'] = uuid.uuid4().hex
        self._assertTokenNotRevoked(third_token)
        self.assertEqual(0, len(revocation_backend.list_events(token=third_token)))
        fourth_token = _sample_blank_token()
        fourth_token['user_id'] = uuid.uuid4().hex
        fourth_token['project_id'] = uuid.uuid4().hex
        fourth_token['audit_id'] = provider.random_urlsafe_str()
        PROVIDERS.revoke_api.revoke(revoke_model.RevokeEvent(project_id=fourth_token['project_id'], audit_id=fourth_token['audit_id']))
        self._assertTokenRevoked(fourth_token)
        self.assertEqual(1, len(revocation_backend.list_events(token=fourth_token)))

    def _user_field_test(self, field_name):
        token = _sample_blank_token()
        token[field_name] = uuid.uuid4().hex
        PROVIDERS.revoke_api.revoke_by_user(user_id=token[field_name])
        self._assertTokenRevoked(token)
        token2 = _sample_blank_token()
        token2[field_name] = uuid.uuid4().hex
        self._assertTokenNotRevoked(token2)

    def test_revoke_by_user(self):
        self._user_field_test('user_id')

    def test_revoke_by_user_matches_trustee(self):
        self._user_field_test('trustee_id')

    def test_revoke_by_user_matches_trustor(self):
        self._user_field_test('trustor_id')

    def test_by_domain_user(self):
        revocation_backend = sql.Revoke()
        user_id = uuid.uuid4().hex
        domain_id = uuid.uuid4().hex
        token_data = _sample_blank_token()
        token_data['user_id'] = user_id
        token_data['identity_domain_id'] = domain_id
        self._assertTokenNotRevoked(token_data)
        self.assertEqual(0, len(revocation_backend.list_events(token=token_data)))
        PROVIDERS.revoke_api.revoke(revoke_model.RevokeEvent(domain_id=domain_id))
        self._assertTokenRevoked(token_data)
        self.assertEqual(1, len(revocation_backend.list_events(token=token_data)))

    def test_by_domain_project(self):
        revocation_backend = sql.Revoke()
        token_data = _sample_blank_token()
        token_data['user_id'] = uuid.uuid4().hex
        token_data['identity_domain_id'] = uuid.uuid4().hex
        token_data['project_id'] = uuid.uuid4().hex
        token_data['assignment_domain_id'] = uuid.uuid4().hex
        self._assertTokenNotRevoked(token_data)
        self.assertEqual(0, len(revocation_backend.list_events(token=token_data)))
        PROVIDERS.revoke_api.revoke(revoke_model.RevokeEvent(domain_id=token_data['assignment_domain_id']))
        self._assertTokenRevoked(token_data)
        self.assertEqual(1, len(revocation_backend.list_events(token=token_data)))

    def test_by_domain_domain(self):
        revocation_backend = sql.Revoke()
        token_data = _sample_blank_token()
        token_data['user_id'] = uuid.uuid4().hex
        token_data['identity_domain_id'] = uuid.uuid4().hex
        token_data['assignment_domain_id'] = uuid.uuid4().hex
        self._assertTokenNotRevoked(token_data)
        self.assertEqual(0, len(revocation_backend.list_events(token=token_data)))
        PROVIDERS.revoke_api.revoke(revoke_model.RevokeEvent(domain_id=token_data['assignment_domain_id']))
        self._assertTokenRevoked(token_data)
        self.assertEqual(1, len(revocation_backend.list_events(token=token_data)))

    def test_revoke_by_audit_id(self):
        token = _sample_blank_token()
        token['audit_id'] = uuid.uuid4().hex
        token['audit_chain_id'] = token['audit_id']
        PROVIDERS.revoke_api.revoke_by_audit_id(audit_id=token['audit_id'])
        self._assertTokenRevoked(token)
        token2 = _sample_blank_token()
        token2['audit_id'] = uuid.uuid4().hex
        token2['audit_chain_id'] = token2['audit_id']
        self._assertTokenNotRevoked(token2)

    def test_revoke_by_audit_chain_id(self):
        revocation_backend = sql.Revoke()
        audit_id = provider.random_urlsafe_str()
        token = _sample_blank_token()
        token['audit_id'] = audit_id
        token['audit_chain_id'] = audit_id
        self._assertTokenNotRevoked(token)
        self.assertEqual(0, len(revocation_backend.list_events(token=token)))
        PROVIDERS.revoke_api.revoke_by_audit_chain_id(audit_id)
        self._assertTokenRevoked(token)
        self.assertEqual(1, len(revocation_backend.list_events(token=token)))

    @mock.patch.object(timeutils, 'utcnow')
    def test_expired_events_are_removed(self, mock_utcnow):

        def _sample_token_values():
            token = _sample_blank_token()
            token['expires_at'] = utils.isotime(_future_time(), subsecond=True)
            return token
        now = datetime.datetime.utcnow()
        now_plus_2h = now + datetime.timedelta(hours=2)
        mock_utcnow.return_value = now
        token_values = _sample_token_values()
        audit_chain_id = uuid.uuid4().hex
        PROVIDERS.revoke_api.revoke_by_audit_chain_id(audit_chain_id)
        token_values['audit_chain_id'] = audit_chain_id
        self.assertRaises(exception.TokenNotFound, PROVIDERS.revoke_api.check_token, token_values)
        mock_utcnow.return_value = now_plus_2h
        PROVIDERS.revoke_api.revoke_by_audit_chain_id(audit_chain_id)
        self.assertRaises(exception.TokenNotFound, PROVIDERS.revoke_api.check_token, token_values)

    def test_delete_group_without_role_does_not_revoke_users(self):
        revocation_backend = sql.Revoke()
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        group1 = unit.new_group_ref(domain_id=domain['id'])
        group1 = PROVIDERS.identity_api.create_group(group1)
        group2 = unit.new_group_ref(domain_id=domain['id'])
        group2 = PROVIDERS.identity_api.create_group(group2)
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        user1 = unit.new_user_ref(domain_id=domain['id'])
        user1 = PROVIDERS.identity_api.create_user(user1)
        user2 = unit.new_user_ref(domain_id=domain['id'])
        user2 = PROVIDERS.identity_api.create_user(user2)
        PROVIDERS.identity_api.add_user_to_group(user_id=user1['id'], group_id=group1['id'])
        PROVIDERS.identity_api.add_user_to_group(user_id=user2['id'], group_id=group1['id'])
        self.assertEqual(2, len(PROVIDERS.identity_api.list_users_in_group(group1['id'])))
        PROVIDERS.identity_api.delete_group(group1['id'])
        self.assertEqual(0, len(revocation_backend.list_events()))
        PROVIDERS.assignment_api.create_grant(group_id=group2['id'], domain_id=domain['id'], role_id=role['id'])
        grants = PROVIDERS.assignment_api.list_role_assignments(role_id=role['id'])
        self.assertThat(grants, matchers.HasLength(1))
        PROVIDERS.identity_api.add_user_to_group(user_id=user1['id'], group_id=group2['id'])
        PROVIDERS.identity_api.add_user_to_group(user_id=user2['id'], group_id=group2['id'])
        self.assertEqual(2, len(PROVIDERS.identity_api.list_users_in_group(group2['id'])))
        PROVIDERS.identity_api.delete_group(group2['id'])
        self.assertEqual(2, len(revocation_backend.list_events()))