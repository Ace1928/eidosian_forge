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
class IdentityTestCase(test_v3.RestfulTestCase):
    """Test users and groups."""

    def setUp(self):
        super(IdentityTestCase, self).setUp()
        self.useFixture(ksfixtures.KeyRepository(self.config_fixture, 'credential', credential_fernet.MAX_ACTIVE_KEYS))
        self.group = unit.new_group_ref(domain_id=self.domain_id)
        self.group = PROVIDERS.identity_api.create_group(self.group)
        self.group_id = self.group['id']
        self.credential = unit.new_credential_ref(user_id=self.user['id'], project_id=self.project_id)
        PROVIDERS.credential_api.create_credential(self.credential['id'], self.credential)

    def test_create_user(self):
        """Call ``POST /users``."""
        ref = unit.new_user_ref(domain_id=self.domain_id)
        r = self.post('/users', body={'user': ref})
        return self.assertValidUserResponse(r, ref)

    def test_create_user_without_domain(self):
        """Call ``POST /users`` without specifying domain.

        According to the identity-api specification, if you do not
        explicitly specific the domain_id in the entity, it should
        take the domain scope of the token as the domain_id.

        """
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        user = unit.create_user(PROVIDERS.identity_api, domain_id=domain['id'])
        PROVIDERS.assignment_api.create_grant(role_id=self.role_id, user_id=user['id'], domain_id=domain['id'])
        ref = unit.new_user_ref(domain_id=domain['id'])
        ref_nd = ref.copy()
        ref_nd.pop('domain_id')
        auth = self.build_authentication_request(user_id=user['id'], password=user['password'], domain_id=domain['id'])
        r = self.post('/users', body={'user': ref_nd}, auth=auth)
        self.assertValidUserResponse(r, ref)
        ref = unit.new_user_ref(domain_id=domain['id'])
        ref_nd = ref.copy()
        ref_nd.pop('domain_id')
        auth = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
        with mock.patch('oslo_log.versionutils.report_deprecated_feature') as mock_dep:
            r = self.post('/users', body={'user': ref_nd}, auth=auth)
            self.assertTrue(mock_dep.called)
        ref['domain_id'] = CONF.identity.default_domain_id
        return self.assertValidUserResponse(r, ref)

    def test_create_user_with_admin_token_and_domain(self):
        """Call ``POST /users`` with admin token and domain id."""
        ref = unit.new_user_ref(domain_id=self.domain_id)
        self.post('/users', body={'user': ref}, token=self.get_admin_token(), expected_status=http.client.CREATED)

    def test_user_management_normalized_keys(self):
        """Illustrate the inconsistent handling of hyphens in keys.

        To quote Morgan in bug 1526244:

            the reason this is converted from "domain-id" to "domain_id" is
            because of how we process/normalize data. The way we have to handle
            specific data types for known columns requires avoiding "-" in the
            actual python code since "-" is not valid for attributes in python
            w/o significant use of "getattr" etc.

            In short, historically we handle some things in conversions. The
            use of "extras" has long been a poor design choice that leads to
            odd/strange inconsistent behaviors because of other choices made in
            handling data from within the body. (In many cases we convert from
            "-" to "_" throughout openstack)

        Source: https://bugs.launchpad.net/keystone/+bug/1526244/comments/9

        """
        domain1 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
        domain2 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain2['id'], domain2)
        user = unit.new_user_ref(domain_id=domain1['id'])
        r = self.post('/users', body={'user': user})
        self.assertValidUserResponse(r, user)
        user['id'] = r.json['user']['id']
        r = self.get('/users?domain-id=%s' % domain1['id'])
        self.assertValidUserListResponse(r, ref=self.user)
        self.assertNotEqual(domain1['id'], self.user['domain_id'])
        user = unit.new_user_ref(domain_id=domain1['id'])
        user['domain-id'] = user.pop('domain_id')
        r = self.post('/users', body={'user': user})
        self.assertNotIn('domain-id', r.json['user'])
        self.assertEqual(domain1['id'], r.json['user']['domain_id'])
        user['domain_id'] = user.pop('domain-id')
        self.assertValidUserResponse(r, user)
        user['id'] = r.json['user']['id']
        r = self.patch('/users/%s' % user['id'], body={'user': {'domain-id': domain2['id']}})
        self.assertEqual(domain2['id'], r.json['user']['domain-id'])
        self.assertEqual(user['domain_id'], r.json['user']['domain_id'])
        self.assertNotEqual(domain2['id'], user['domain_id'])
        self.assertValidUserResponse(r, user)

    def test_create_user_bad_request(self):
        """Call ``POST /users``."""
        self.post('/users', body={'user': {}}, expected_status=http.client.BAD_REQUEST)

    def test_create_user_bad_domain_id(self):
        """Call ``POST /users``."""
        self.post('/users', body={'user': {'name': 'baddomain', 'domain_id': 'DEFaUlT'}}, expected_status=http.client.NOT_FOUND)

    def test_list_head_users(self):
        """Call ``GET & HEAD /users``."""
        resource_url = '/users'
        r = self.get(resource_url)
        self.assertValidUserListResponse(r, ref=self.user, resource_url=resource_url)
        self.head(resource_url, expected_status=http.client.OK)

    def test_list_users_with_multiple_backends(self):
        """Call ``GET /users`` when multiple backends is enabled.

        In this scenario, the controller requires a domain to be specified
        either as a filter or by using a domain scoped token.

        """
        self.config_fixture.config(group='identity', domain_specific_drivers_enabled=True)
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        project = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project['id'], project)
        user = unit.create_user(PROVIDERS.identity_api, domain_id=domain['id'])
        PROVIDERS.assignment_api.create_grant(role_id=self.role_id, user_id=user['id'], domain_id=domain['id'])
        PROVIDERS.assignment_api.create_grant(role_id=self.role_id, user_id=user['id'], project_id=project['id'])
        dom_auth = self.build_authentication_request(user_id=user['id'], password=user['password'], domain_id=domain['id'])
        project_auth = self.build_authentication_request(user_id=user['id'], password=user['password'], project_id=project['id'])
        resource_url = '/users'
        r = self.get(resource_url, auth=dom_auth)
        self.assertValidUserListResponse(r, ref=user, resource_url=resource_url)
        resource_url = '/users'
        r = self.get(resource_url, auth=project_auth)
        self.assertValidUserListResponse(r, ref=user, resource_url=resource_url)
        resource_url = '/users?domain_id=%(domain_id)s' % {'domain_id': domain['id']}
        r = self.get(resource_url)
        self.assertValidUserListResponse(r, ref=user, resource_url=resource_url)

    def test_list_users_no_default_project(self):
        """Call ``GET /users`` making sure no default_project_id."""
        user = unit.new_user_ref(self.domain_id)
        user = PROVIDERS.identity_api.create_user(user)
        resource_url = '/users'
        r = self.get(resource_url)
        self.assertValidUserListResponse(r, ref=user, resource_url=resource_url)

    def test_get_head_user(self):
        """Call ``GET & HEAD /users/{user_id}``."""
        resource_url = '/users/%(user_id)s' % {'user_id': self.user['id']}
        r = self.get(resource_url)
        self.assertValidUserResponse(r, self.user)
        self.head(resource_url, expected_status=http.client.OK)

    def test_get_user_does_not_include_extra_attributes(self):
        """Call ``GET /users/{user_id}`` extra attributes are not included."""
        user = unit.new_user_ref(domain_id=self.domain_id, project_id=self.project_id)
        user = PROVIDERS.identity_api.create_user(user)
        self.assertNotIn('created_at', user)
        self.assertNotIn('last_active_at', user)

    def test_get_user_includes_required_attributes(self):
        """Call ``GET /users/{user_id}`` required attributes are included."""
        user = unit.new_user_ref(domain_id=self.domain_id, project_id=self.project_id)
        user = PROVIDERS.identity_api.create_user(user)
        self.assertIn('id', user)
        self.assertIn('name', user)
        self.assertIn('enabled', user)
        self.assertIn('password_expires_at', user)
        r = self.get('/users/%(user_id)s' % {'user_id': user['id']})
        self.assertValidUserResponse(r, user)

    def test_get_user_with_default_project(self):
        """Call ``GET /users/{user_id}`` making sure of default_project_id."""
        user = unit.new_user_ref(domain_id=self.domain_id, project_id=self.project_id)
        user = PROVIDERS.identity_api.create_user(user)
        r = self.get('/users/%(user_id)s' % {'user_id': user['id']})
        self.assertValidUserResponse(r, user)

    def test_add_user_to_group(self):
        """Call ``PUT /groups/{group_id}/users/{user_id}``."""
        self.put('/groups/%(group_id)s/users/%(user_id)s' % {'group_id': self.group_id, 'user_id': self.user['id']})

    def test_list_head_groups_for_user(self):
        """Call ``GET & HEAD /users/{user_id}/groups``."""
        user1 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
        user2 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
        self.put('/groups/%(group_id)s/users/%(user_id)s' % {'group_id': self.group_id, 'user_id': user1['id']})
        auth = self.build_authentication_request(user_id=user1['id'], password=user1['password'])
        resource_url = '/users/%(user_id)s/groups' % {'user_id': user1['id']}
        r = self.get(resource_url, auth=auth)
        self.assertValidGroupListResponse(r, ref=self.group, resource_url=resource_url)
        self.head(resource_url, auth=auth, expected_status=http.client.OK)
        resource_url = '/users/%(user_id)s/groups' % {'user_id': user1['id']}
        r = self.get(resource_url)
        self.assertValidGroupListResponse(r, ref=self.group, resource_url=resource_url)
        self.head(resource_url, expected_status=http.client.OK)
        auth = self.build_authentication_request(user_id=user2['id'], password=user2['password'])
        resource_url = '/users/%(user_id)s/groups' % {'user_id': user1['id']}
        self.get(resource_url, auth=auth, expected_status=exception.ForbiddenAction.code)
        self.head(resource_url, auth=auth, expected_status=exception.ForbiddenAction.code)

    def test_check_user_in_group(self):
        """Call ``HEAD /groups/{group_id}/users/{user_id}``."""
        self.put('/groups/%(group_id)s/users/%(user_id)s' % {'group_id': self.group_id, 'user_id': self.user['id']})
        self.head('/groups/%(group_id)s/users/%(user_id)s' % {'group_id': self.group_id, 'user_id': self.user['id']})

    def test_list_head_users_in_group(self):
        """Call ``GET & HEAD /groups/{group_id}/users``."""
        self.put('/groups/%(group_id)s/users/%(user_id)s' % {'group_id': self.group_id, 'user_id': self.user['id']})
        resource_url = '/groups/%(group_id)s/users' % {'group_id': self.group_id}
        r = self.get(resource_url)
        self.assertValidUserListResponse(r, ref=self.user, resource_url=resource_url)
        self.assertIn('/groups/%(group_id)s/users' % {'group_id': self.group_id}, r.result['links']['self'])
        self.head(resource_url, expected_status=http.client.OK)

    def test_remove_user_from_group(self):
        """Call ``DELETE /groups/{group_id}/users/{user_id}``."""
        self.put('/groups/%(group_id)s/users/%(user_id)s' % {'group_id': self.group_id, 'user_id': self.user['id']})
        self.delete('/groups/%(group_id)s/users/%(user_id)s' % {'group_id': self.group_id, 'user_id': self.user['id']})

    def test_update_ephemeral_user(self):
        federated_user_a = model.FederatedUser()
        federated_user_b = model.FederatedUser()
        federated_user_a.idp_id = 'a_idp'
        federated_user_b.idp_id = 'b_idp'
        federated_user_a.display_name = 'federated_a'
        federated_user_b.display_name = 'federated_b'
        federated_users = [federated_user_a, federated_user_b]
        user_a = model.User()
        user_a.federated_users = federated_users
        self.assertEqual(federated_user_a.display_name, user_a.name)
        self.assertIsNone(user_a.password)
        user_a.name = 'new_federated_a'
        self.assertEqual('new_federated_a', user_a.name)
        self.assertIsNone(user_a.local_user)

    def test_update_user(self):
        """Call ``PATCH /users/{user_id}``."""
        user = unit.new_user_ref(domain_id=self.domain_id)
        del user['id']
        r = self.patch('/users/%(user_id)s' % {'user_id': self.user['id']}, body={'user': user})
        self.assertValidUserResponse(r, user)

    def test_admin_password_reset(self):
        user_ref = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
        old_password_auth = self.build_authentication_request(user_id=user_ref['id'], password=user_ref['password'])
        r = self.v3_create_token(old_password_auth)
        old_token = r.headers.get('X-Subject-Token')
        old_token_auth = self.build_authentication_request(token=old_token)
        self.v3_create_token(old_token_auth)
        new_password = uuid.uuid4().hex
        self.patch('/users/%s' % user_ref['id'], body={'user': {'password': new_password}})
        self.v3_create_token(old_password_auth, expected_status=http.client.UNAUTHORIZED)
        self.v3_create_token(old_token_auth, expected_status=http.client.NOT_FOUND)
        new_password_auth = self.build_authentication_request(user_id=user_ref['id'], password=new_password)
        self.v3_create_token(new_password_auth)

    def test_admin_password_reset_with_min_password_age_enabled(self):
        self.config_fixture.config(group='security_compliance', minimum_password_age=1)
        user_ref = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
        new_password = uuid.uuid4().hex
        r = self.patch('/users/%s' % user_ref['id'], body={'user': {'password': new_password}})
        self.assertValidUserResponse(r, user_ref)
        new_password_auth = self.build_authentication_request(user_id=user_ref['id'], password=new_password)
        self.v3_create_token(new_password_auth)

    def test_admin_password_reset_with_password_lock(self):
        user_ref = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
        lock_pw_opt = options.LOCK_PASSWORD_OPT.option_name
        update_user_body = {'user': {'options': {lock_pw_opt: True}}}
        self.patch('/users/%s' % user_ref['id'], body=update_user_body)
        new_password = uuid.uuid4().hex
        r = self.patch('/users/%s' % user_ref['id'], body={'user': {'password': new_password}})
        self.assertValidUserResponse(r, user_ref)
        new_password_auth = self.build_authentication_request(user_id=user_ref['id'], password=new_password)
        self.v3_create_token(new_password_auth)

    def test_update_user_domain_id(self):
        """Call ``PATCH /users/{user_id}`` with domain_id.

        A user's `domain_id` is immutable. Ensure that any attempts to update
        the `domain_id` of a user fails.
        """
        user = unit.new_user_ref(domain_id=self.domain['id'])
        user = PROVIDERS.identity_api.create_user(user)
        user['domain_id'] = CONF.identity.default_domain_id
        self.patch('/users/%(user_id)s' % {'user_id': user['id']}, body={'user': user}, expected_status=exception.ValidationError.code)

    def test_delete_user(self):
        """Call ``DELETE /users/{user_id}``.

        As well as making sure the delete succeeds, we ensure
        that any credentials that reference this user are
        also deleted, while other credentials are unaffected.
        In addition, no tokens should remain valid for this user.

        """
        r = PROVIDERS.credential_api.get_credential(self.credential['id'])
        self.assertDictEqual(self.credential, r)
        user2 = unit.new_user_ref(domain_id=self.domain['id'], project_id=self.project['id'])
        user2 = PROVIDERS.identity_api.create_user(user2)
        credential2 = unit.new_credential_ref(user_id=user2['id'], project_id=self.project['id'])
        PROVIDERS.credential_api.create_credential(credential2['id'], credential2)
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
        token = self.get_requested_token(auth_data)
        self.head('/auth/tokens', headers={'X-Subject-Token': token}, expected_status=http.client.OK)
        self.delete('/users/%(user_id)s' % {'user_id': self.user['id']})
        self.assertRaises(exception.CredentialNotFound, PROVIDERS.credential_api.get_credential, self.credential['id'])
        r = PROVIDERS.credential_api.get_credential(credential2['id'])
        self.assertDictEqual(credential2, r)

    def test_delete_user_retries_on_deadlock(self):
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
        user_ref = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
        try:
            PROVIDERS.identity_api.delete_user(user_id=user_ref['id'])
        finally:
            if side_effect.patched:
                patcher.stop()
        call_count = sql_delete_mock.call_count
        delete_user_attempt_count = 2
        self.assertEqual(call_count, delete_user_attempt_count)

    def test_create_group(self):
        """Call ``POST /groups``."""
        ref = unit.new_group_ref(domain_id=self.domain_id)
        r = self.post('/groups', body={'group': ref})
        return self.assertValidGroupResponse(r, ref)

    def test_create_group_bad_request(self):
        """Call ``POST /groups``."""
        self.post('/groups', body={'group': {}}, expected_status=http.client.BAD_REQUEST)

    def test_list_head_groups(self):
        """Call ``GET & HEAD /groups``."""
        resource_url = '/groups'
        r = self.get(resource_url)
        self.assertValidGroupListResponse(r, ref=self.group, resource_url=resource_url)
        self.head(resource_url, expected_status=http.client.OK)

    def test_get_head_group(self):
        """Call ``GET & HEAD /groups/{group_id}``."""
        resource_url = '/groups/%(group_id)s' % {'group_id': self.group_id}
        r = self.get(resource_url)
        self.assertValidGroupResponse(r, self.group)
        self.head(resource_url, expected_status=http.client.OK)

    def test_update_group(self):
        """Call ``PATCH /groups/{group_id}``."""
        group = unit.new_group_ref(domain_id=self.domain_id)
        del group['id']
        r = self.patch('/groups/%(group_id)s' % {'group_id': self.group_id}, body={'group': group})
        self.assertValidGroupResponse(r, group)

    def test_update_group_domain_id(self):
        """Call ``PATCH /groups/{group_id}`` with domain_id.

        A group's `domain_id` is immutable. Ensure that any attempts to update
        the `domain_id` of a group fails.
        """
        self.group['domain_id'] = CONF.identity.default_domain_id
        self.patch('/groups/%(group_id)s' % {'group_id': self.group['id']}, body={'group': self.group}, expected_status=exception.ValidationError.code)

    def test_delete_group(self):
        """Call ``DELETE /groups/{group_id}``."""
        self.delete('/groups/%(group_id)s' % {'group_id': self.group_id})

    def test_create_user_password_not_logged(self):
        log_fix = self.useFixture(fixtures.FakeLogger(level=log.DEBUG))
        ref = unit.new_user_ref(domain_id=self.domain_id)
        self.post('/users', body={'user': ref})
        self.assertNotIn(ref['password'], log_fix.output)

    def test_update_password_not_logged(self):
        log_fix = self.useFixture(fixtures.FakeLogger(level=log.DEBUG))
        user_ref = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
        self.assertNotIn(user_ref['password'], log_fix.output)
        new_password = uuid.uuid4().hex
        self.patch('/users/%s' % user_ref['id'], body={'user': {'password': new_password}})
        self.assertNotIn(new_password, log_fix.output)

    def test_setting_default_project_id_to_domain_failed(self):
        """Call ``POST and PATCH /users`` default_project_id=domain_id.

        Make sure we validate the default_project_id if it is specified.
        It cannot be set to a domain_id, even for a project acting as domain
        right now. That's because we haven't sort out the issuing
        project-scoped token for project acting as domain bit yet. Once we
        got that sorted out, we can relax this constraint.

        """
        ref = unit.new_user_ref(domain_id=self.domain_id, project_id=self.domain_id)
        self.post('/users', body={'user': ref}, token=CONF.admin_token, expected_status=http.client.BAD_REQUEST)
        user = {'default_project_id': self.domain_id}
        self.patch('/users/%(user_id)s' % {'user_id': self.user['id']}, body={'user': user}, token=CONF.admin_token, expected_status=http.client.BAD_REQUEST)