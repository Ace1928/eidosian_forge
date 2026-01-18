import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
class AssignmentTestCase(test_v3.RestfulTestCase, test_v3.AssignmentTestMixin, SystemRoleAssignmentMixin):
    """Test roles and role assignments."""

    def setUp(self):
        super(AssignmentTestCase, self).setUp()
        self.group = unit.new_group_ref(domain_id=self.domain_id)
        self.group = PROVIDERS.identity_api.create_group(self.group)
        self.group_id = self.group['id']

    def test_create_role(self):
        """Call ``POST /roles``."""
        ref = unit.new_role_ref()
        r = self.post('/roles', body={'role': ref})
        return self.assertValidRoleResponse(r, ref)

    def test_create_role_bad_request(self):
        """Call ``POST /roles``."""
        self.post('/roles', body={'role': {}}, expected_status=http.client.BAD_REQUEST)

    def test_list_head_roles(self):
        """Call ``GET & HEAD /roles``."""
        resource_url = '/roles'
        r = self.get(resource_url)
        self.assertValidRoleListResponse(r, ref=self.role, resource_url=resource_url)
        self.head(resource_url, expected_status=http.client.OK)

    def test_get_head_role(self):
        """Call ``GET & HEAD /roles/{role_id}``."""
        resource_url = '/roles/%(role_id)s' % {'role_id': self.role_id}
        r = self.get(resource_url)
        self.assertValidRoleResponse(r, self.role)
        self.head(resource_url, expected_status=http.client.OK)

    def test_update_role(self):
        """Call ``PATCH /roles/{role_id}``."""
        ref = unit.new_role_ref()
        del ref['id']
        r = self.patch('/roles/%(role_id)s' % {'role_id': self.role_id}, body={'role': ref})
        self.assertValidRoleResponse(r, ref)

    def test_delete_role(self):
        """Call ``DELETE /roles/{role_id}``."""
        self.delete('/roles/%(role_id)s' % {'role_id': self.role_id})

    def test_crud_user_project_role_grants(self):
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        collection_url = '/projects/%(project_id)s/users/%(user_id)s/roles' % {'project_id': self.project['id'], 'user_id': self.user['id']}
        member_url = '%(collection_url)s/%(role_id)s' % {'collection_url': collection_url, 'role_id': role['id']}
        r = self.get(collection_url)
        self.assertValidRoleListResponse(r, ref=self.role, expected_length=1)
        self.put(member_url)
        self.head(member_url)
        self.get(member_url, expected_status=http.client.NO_CONTENT)
        r = self.get(collection_url)
        self.assertValidRoleListResponse(r, ref=role, resource_url=collection_url, expected_length=2)
        self.head(collection_url, expected_status=http.client.OK)
        self.delete(member_url)
        r = self.get(collection_url)
        self.assertValidRoleListResponse(r, ref=self.role, expected_length=1)
        self.assertIn(collection_url, r.result['links']['self'])
        self.head(collection_url, expected_status=http.client.OK)

    def test_crud_user_project_role_grants_no_user(self):
        """Grant role on a project to a user that doesn't exist.

        When grant a role on a project to a user that doesn't exist, the server
        returns Not Found for the user.

        """
        user_id = uuid.uuid4().hex
        collection_url = '/projects/%(project_id)s/users/%(user_id)s/roles' % {'project_id': self.project['id'], 'user_id': user_id}
        member_url = '%(collection_url)s/%(role_id)s' % {'collection_url': collection_url, 'role_id': self.role_id}
        self.put(member_url, expected_status=http.client.NOT_FOUND)
        self.head(member_url, expected_status=http.client.NOT_FOUND)
        self.get(member_url, expected_status=http.client.NOT_FOUND)

    def test_crud_user_domain_role_grants(self):
        time = datetime.datetime.utcnow()
        with freezegun.freeze_time(time) as frozen_datetime:
            collection_url = '/domains/%(domain_id)s/users/%(user_id)s/roles' % {'domain_id': self.domain_id, 'user_id': self.user['id']}
            member_url = '%(collection_url)s/%(role_id)s' % {'collection_url': collection_url, 'role_id': self.role_id}
            self.put(member_url)
            self.head(member_url)
            self.get(member_url, expected_status=http.client.NO_CONTENT)
            r = self.get(collection_url)
            self.assertValidRoleListResponse(r, ref=self.role, resource_url=collection_url)
            self.head(collection_url, expected_status=http.client.OK)
            self.delete(member_url)
            frozen_datetime.tick(delta=datetime.timedelta(seconds=1))
            r = self.get(collection_url)
            self.assertValidRoleListResponse(r, expected_length=0, resource_url=collection_url)
            self.head(collection_url, expected_status=http.client.OK)

    def test_crud_user_domain_role_grants_no_user(self):
        """Grant role on a domain to a user that doesn't exist.

        When grant a role on a domain to a user that doesn't exist, the server
        returns 404 Not Found for the user.

        """
        user_id = uuid.uuid4().hex
        collection_url = '/domains/%(domain_id)s/users/%(user_id)s/roles' % {'domain_id': self.domain_id, 'user_id': user_id}
        member_url = '%(collection_url)s/%(role_id)s' % {'collection_url': collection_url, 'role_id': self.role_id}
        self.put(member_url, expected_status=http.client.NOT_FOUND)
        self.head(member_url, expected_status=http.client.NOT_FOUND)
        self.get(member_url, expected_status=http.client.NOT_FOUND)

    def test_crud_group_project_role_grants(self):
        time = datetime.datetime.utcnow()
        with freezegun.freeze_time(time) as frozen_datetime:
            collection_url = '/projects/%(project_id)s/groups/%(group_id)s/roles' % {'project_id': self.project_id, 'group_id': self.group_id}
            member_url = '%(collection_url)s/%(role_id)s' % {'collection_url': collection_url, 'role_id': self.role_id}
            self.put(member_url)
            self.head(member_url)
            self.get(member_url, expected_status=http.client.NO_CONTENT)
            r = self.get(collection_url)
            self.assertValidRoleListResponse(r, ref=self.role, resource_url=collection_url)
            self.head(collection_url, expected_status=http.client.OK)
            self.delete(member_url)
            frozen_datetime.tick(delta=datetime.timedelta(seconds=1))
            r = self.get(collection_url)
            self.assertValidRoleListResponse(r, expected_length=0, resource_url=collection_url)
            self.head(collection_url, expected_status=http.client.OK)

    def test_crud_group_project_role_grants_no_group(self):
        """Grant role on a project to a group that doesn't exist.

        When grant a role on a project to a group that doesn't exist, the
        server returns 404 Not Found for the group.

        """
        group_id = uuid.uuid4().hex
        collection_url = '/projects/%(project_id)s/groups/%(group_id)s/roles' % {'project_id': self.project_id, 'group_id': group_id}
        member_url = '%(collection_url)s/%(role_id)s' % {'collection_url': collection_url, 'role_id': self.role_id}
        self.put(member_url, expected_status=http.client.NOT_FOUND)
        self.head(member_url, expected_status=http.client.NOT_FOUND)
        self.get(member_url, expected_status=http.client.NOT_FOUND)

    def test_crud_group_domain_role_grants(self):
        time = datetime.datetime.utcnow()
        with freezegun.freeze_time(time) as frozen_datetime:
            collection_url = '/domains/%(domain_id)s/groups/%(group_id)s/roles' % {'domain_id': self.domain_id, 'group_id': self.group_id}
            member_url = '%(collection_url)s/%(role_id)s' % {'collection_url': collection_url, 'role_id': self.role_id}
            self.put(member_url)
            self.head(member_url)
            self.get(member_url, expected_status=http.client.NO_CONTENT)
            r = self.get(collection_url)
            self.assertValidRoleListResponse(r, ref=self.role, resource_url=collection_url)
            self.head(collection_url, expected_status=http.client.OK)
            self.delete(member_url)
            frozen_datetime.tick(delta=datetime.timedelta(seconds=1))
            r = self.get(collection_url)
            self.assertValidRoleListResponse(r, expected_length=0, resource_url=collection_url)
            self.head(collection_url, expected_status=http.client.OK)

    def test_crud_group_domain_role_grants_no_group(self):
        """Grant role on a domain to a group that doesn't exist.

        When grant a role on a domain to a group that doesn't exist, the server
        returns 404 Not Found for the group.

        """
        group_id = uuid.uuid4().hex
        collection_url = '/domains/%(domain_id)s/groups/%(group_id)s/roles' % {'domain_id': self.domain_id, 'group_id': group_id}
        member_url = '%(collection_url)s/%(role_id)s' % {'collection_url': collection_url, 'role_id': self.role_id}
        self.put(member_url, expected_status=http.client.NOT_FOUND)
        self.head(member_url, expected_status=http.client.NOT_FOUND)
        self.get(member_url, expected_status=http.client.NOT_FOUND)

    def _create_new_user_and_assign_role_on_project(self):
        """Create a new user and assign user a role on a project."""
        new_user = unit.new_user_ref(domain_id=self.domain_id)
        user_ref = PROVIDERS.identity_api.create_user(new_user)
        collection_url = '/projects/%(project_id)s/users/%(user_id)s/roles' % {'project_id': self.project_id, 'user_id': user_ref['id']}
        member_url = '%(collection_url)s/%(role_id)s' % {'collection_url': collection_url, 'role_id': self.role_id}
        self.put(member_url)
        self.head(member_url)
        self.get(member_url, expected_status=http.client.NO_CONTENT)
        return (member_url, user_ref)

    def test_delete_user_before_removing_role_assignment_succeeds(self):
        """Call ``DELETE`` on the user before the role assignment."""
        member_url, user = self._create_new_user_and_assign_role_on_project()
        PROVIDERS.identity_api.driver.delete_user(user['id'])
        self.delete(member_url)
        self.head(member_url, expected_status=http.client.NOT_FOUND)

    def test_delete_group_before_removing_role_assignment_succeeds(self):
        self.config_fixture.config(group='cache', enabled=False)
        group = unit.new_group_ref(domain_id=self.domain_id)
        group_ref = PROVIDERS.identity_api.create_group(group)
        collection_url = '/projects/%(project_id)s/groups/%(group_id)s/roles' % {'project_id': self.project_id, 'group_id': group_ref['id']}
        member_url = '%(collection_url)s/%(role_id)s' % {'collection_url': collection_url, 'role_id': self.role_id}
        self.put(member_url)
        self.head(member_url)
        self.get(member_url, expected_status=http.client.NO_CONTENT)
        PROVIDERS.identity_api.driver.delete_group(group_ref['id'])
        self.delete(member_url)

    def test_delete_user_before_removing_system_assignments_succeeds(self):
        system_role = self._create_new_role()
        user = self._create_user()
        path = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': user['id'], 'role_id': system_role}
        self.put(path)
        response = self.get('/role_assignments')
        number_of_assignments = len(response.json_body['role_assignments'])
        path = '/users/%(user_id)s' % {'user_id': user['id']}
        self.delete(path)
        response = self.get('/role_assignments')
        self.assertValidRoleAssignmentListResponse(response, expected_length=number_of_assignments - 1)

    def test_delete_user_and_check_role_assignment_fails(self):
        """Call ``DELETE`` on the user and check the role assignment."""
        member_url, user = self._create_new_user_and_assign_role_on_project()
        PROVIDERS.identity_api.delete_user(user['id'])
        self.head(member_url, expected_status=http.client.NOT_FOUND)

    def test_token_revoked_once_group_role_grant_revoked(self):
        """Test token invalid when direct & indirect role on user is revoked.

        When a role granted to a group is revoked for a given scope,
        and user direct role is revoked, then tokens created
        by user will be invalid.

        """
        time = datetime.datetime.utcnow()
        with freezegun.freeze_time(time) as frozen_datetime:
            PROVIDERS.assignment_api.create_grant(role_id=self.role['id'], project_id=self.project['id'], group_id=self.group['id'])
            PROVIDERS.identity_api.add_user_to_group(user_id=self.user['id'], group_id=self.group['id'])
            auth_body = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
            token_resp = self.post('/auth/tokens', body=auth_body)
            token = token_resp.headers.get('x-subject-token')
            self.head('/auth/tokens', headers={'x-subject-token': token}, expected_status=http.client.OK)
            frozen_datetime.tick(delta=datetime.timedelta(seconds=1))
            PROVIDERS.assignment_api.delete_grant(role_id=self.role['id'], project_id=self.project['id'], group_id=self.group['id'])
            PROVIDERS.assignment_api.delete_grant(role_id=self.role['id'], project_id=self.project['id'], user_id=self.user['id'])
            frozen_datetime.tick(delta=datetime.timedelta(seconds=1))
            self.head('/auth/tokens', token=token, expected_status=http.client.UNAUTHORIZED)

    def test_delete_group_before_removing_system_assignments_succeeds(self):
        system_role = self._create_new_role()
        group = self._create_group()
        path = '/system/groups/%(group_id)s/roles/%(role_id)s' % {'group_id': group['id'], 'role_id': system_role}
        self.put(path)
        response = self.get('/role_assignments')
        number_of_assignments = len(response.json_body['role_assignments'])
        path = '/groups/%(group_id)s' % {'group_id': group['id']}
        self.delete(path)
        response = self.get('/role_assignments')
        self.assertValidRoleAssignmentListResponse(response, expected_length=number_of_assignments - 1)

    @unit.skip_if_cache_disabled('assignment')
    def test_delete_grant_from_user_and_project_invalidate_cache(self):
        new_project = unit.new_project_ref(domain_id=self.domain_id)
        PROVIDERS.resource_api.create_project(new_project['id'], new_project)
        collection_url = '/projects/%(project_id)s/users/%(user_id)s/roles' % {'project_id': new_project['id'], 'user_id': self.user['id']}
        member_url = '%(collection_url)s/%(role_id)s' % {'collection_url': collection_url, 'role_id': self.role_id}
        self.put(member_url)
        self.head(member_url)
        self.get(member_url, expected_status=http.client.NO_CONTENT)
        resp = self.get(collection_url)
        self.assertValidRoleListResponse(resp, ref=self.role, resource_url=collection_url)
        self.delete(member_url)
        resp = self.get(collection_url)
        self.assertListEqual(resp.json_body['roles'], [])

    @unit.skip_if_cache_disabled('assignment')
    def test_delete_grant_from_user_and_domain_invalidates_cache(self):
        new_domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(new_domain['id'], new_domain)
        collection_url = '/domains/%(domain_id)s/users/%(user_id)s/roles' % {'domain_id': new_domain['id'], 'user_id': self.user['id']}
        member_url = '%(collection_url)s/%(role_id)s' % {'collection_url': collection_url, 'role_id': self.role_id}
        self.put(member_url)
        self.head(member_url)
        self.get(member_url, expected_status=http.client.NO_CONTENT)
        resp = self.get(collection_url)
        self.assertValidRoleListResponse(resp, ref=self.role, resource_url=collection_url)
        self.delete(member_url)
        resp = self.get(collection_url)
        self.assertListEqual(resp.json_body['roles'], [])

    @unit.skip_if_cache_disabled('assignment')
    def test_delete_grant_from_group_and_project_invalidates_cache(self):
        new_project = unit.new_project_ref(domain_id=self.domain_id)
        PROVIDERS.resource_api.create_project(new_project['id'], new_project)
        collection_url = '/projects/%(project_id)s/groups/%(group_id)s/roles' % {'project_id': new_project['id'], 'group_id': self.group['id']}
        member_url = '%(collection_url)s/%(role_id)s' % {'collection_url': collection_url, 'role_id': self.role_id}
        self.put(member_url)
        self.head(member_url)
        self.get(member_url, expected_status=http.client.NO_CONTENT)
        resp = self.get(collection_url)
        self.assertValidRoleListResponse(resp, ref=self.role, resource_url=collection_url)
        self.delete(member_url)
        resp = self.get(collection_url)
        self.assertListEqual(resp.json_body['roles'], [])

    @unit.skip_if_cache_disabled('assignment')
    def test_delete_grant_from_group_and_domain_invalidates_cache(self):
        new_domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(new_domain['id'], new_domain)
        collection_url = '/domains/%(domain_id)s/groups/%(group_id)s/roles' % {'domain_id': new_domain['id'], 'group_id': self.group['id']}
        member_url = '%(collection_url)s/%(role_id)s' % {'collection_url': collection_url, 'role_id': self.role_id}
        self.put(member_url)
        self.head(member_url)
        self.get(member_url, expected_status=http.client.NO_CONTENT)
        resp = self.get(collection_url)
        self.assertValidRoleListResponse(resp, ref=self.role, resource_url=collection_url)
        self.delete(member_url)
        resp = self.get(collection_url)
        self.assertListEqual(resp.json_body['roles'], [])

    def test_get_head_role_assignments(self):
        """Call ``GET & HEAD /role_assignments``.

        The sample data set up already has a user, group and project
        that is part of self.domain. We use these plus a new user
        we create as our data set, making sure we ignore any
        role assignments that are already in existence.

        Since we don't yet support a first class entity for role
        assignments, we are only testing the LIST API.  To create
        and delete the role assignments we use the old grant APIs.

        Test Plan:

        - Create extra user for tests
        - Get a list of all existing role assignments
        - Add a new assignment for each of the four combinations, i.e.
          group+domain, user+domain, group+project, user+project, using
          the same role each time
        - Get a new list of all role assignments, checking these four new
          ones have been added
        - Then delete the four we added
        - Get a new list of all role assignments, checking the four have
          been removed

        """
        time = datetime.datetime.utcnow()
        with freezegun.freeze_time(time) as frozen_datetime:
            user1 = unit.new_user_ref(domain_id=self.domain['id'])
            user1 = PROVIDERS.identity_api.create_user(user1)
            role = unit.new_role_ref()
            PROVIDERS.role_api.create_role(role['id'], role)
            collection_url = '/role_assignments'
            r = self.get(collection_url)
            self.assertValidRoleAssignmentListResponse(r, resource_url=collection_url)
            self.head(collection_url, expected_status=http.client.OK)
            existing_assignments = len(r.result.get('role_assignments'))
            gd_entity = self.build_role_assignment_entity(domain_id=self.domain_id, group_id=self.group_id, role_id=role['id'])
            self.put(gd_entity['links']['assignment'])
            r = self.get(collection_url)
            self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments + 1, resource_url=collection_url)
            self.assertRoleAssignmentInListResponse(r, gd_entity)
            self.head(collection_url, expected_status=http.client.OK)
            ud_entity = self.build_role_assignment_entity(domain_id=self.domain_id, user_id=user1['id'], role_id=role['id'])
            self.put(ud_entity['links']['assignment'])
            r = self.get(collection_url)
            self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments + 2, resource_url=collection_url)
            self.assertRoleAssignmentInListResponse(r, ud_entity)
            self.head(collection_url, expected_status=http.client.OK)
            gp_entity = self.build_role_assignment_entity(project_id=self.project_id, group_id=self.group_id, role_id=role['id'])
            self.put(gp_entity['links']['assignment'])
            r = self.get(collection_url)
            self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments + 3, resource_url=collection_url)
            self.assertRoleAssignmentInListResponse(r, gp_entity)
            self.head(collection_url, expected_status=http.client.OK)
            up_entity = self.build_role_assignment_entity(project_id=self.project_id, user_id=user1['id'], role_id=role['id'])
            self.put(up_entity['links']['assignment'])
            r = self.get(collection_url)
            self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments + 4, resource_url=collection_url)
            self.assertRoleAssignmentInListResponse(r, up_entity)
            self.head(collection_url, expected_status=http.client.OK)
            self.delete(gd_entity['links']['assignment'])
            self.delete(ud_entity['links']['assignment'])
            self.delete(gp_entity['links']['assignment'])
            self.delete(up_entity['links']['assignment'])
            frozen_datetime.tick(delta=datetime.timedelta(seconds=1))
            r = self.get(collection_url)
            self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments, resource_url=collection_url)
            self.assertRoleAssignmentNotInListResponse(r, gd_entity)
            self.assertRoleAssignmentNotInListResponse(r, ud_entity)
            self.assertRoleAssignmentNotInListResponse(r, gp_entity)
            self.assertRoleAssignmentNotInListResponse(r, up_entity)
            self.head(collection_url, expected_status=http.client.OK)

    def test_get_effective_role_assignments(self):
        """Call ``GET /role_assignments?effective``.

        Test Plan:

        - Create two extra user for tests
        - Add these users to a group
        - Add a role assignment for the group on a domain
        - Get a list of all role assignments, checking one has been added
        - Then get a list of all effective role assignments - the group
          assignment should have turned into assignments on the domain
          for each of the group members.

        """
        user1 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
        user2 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
        PROVIDERS.identity_api.add_user_to_group(user1['id'], self.group['id'])
        PROVIDERS.identity_api.add_user_to_group(user2['id'], self.group['id'])
        collection_url = '/role_assignments'
        r = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(r, resource_url=collection_url)
        existing_assignments = len(r.result.get('role_assignments'))
        gd_entity = self.build_role_assignment_entity(domain_id=self.domain_id, group_id=self.group_id, role_id=self.role_id)
        self.put(gd_entity['links']['assignment'])
        r = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments + 1, resource_url=collection_url)
        self.assertRoleAssignmentInListResponse(r, gd_entity)
        collection_url = '/role_assignments?effective'
        r = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments + 2, resource_url=collection_url)
        ud_entity = self.build_role_assignment_entity(link=gd_entity['links']['assignment'], domain_id=self.domain_id, user_id=user1['id'], role_id=self.role_id)
        self.assertRoleAssignmentInListResponse(r, ud_entity)
        ud_entity = self.build_role_assignment_entity(link=gd_entity['links']['assignment'], domain_id=self.domain_id, user_id=user2['id'], role_id=self.role_id)
        self.assertRoleAssignmentInListResponse(r, ud_entity)

    def test_check_effective_values_for_role_assignments(self):
        """Call ``GET & HEAD /role_assignments?effective=value``.

        Check the various ways of specifying the 'effective'
        query parameter.  If the 'effective' query parameter
        is included then this should always be treated as meaning 'True'
        unless it is specified as:

        {url}?effective=0

        This is by design to match the agreed way of handling
        policy checking on query/filter parameters.

        Test Plan:

        - Create two extra user for tests
        - Add these users to a group
        - Add a role assignment for the group on a domain
        - Get a list of all role assignments, checking one has been added
        - Then issue various request with different ways of defining
          the 'effective' query parameter. As we have tested the
          correctness of the data coming back when we get effective roles
          in other tests, here we just use the count of entities to
          know if we are getting effective roles or not

        """
        user1 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
        user2 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
        PROVIDERS.identity_api.add_user_to_group(user1['id'], self.group['id'])
        PROVIDERS.identity_api.add_user_to_group(user2['id'], self.group['id'])
        collection_url = '/role_assignments'
        r = self.get(collection_url, expected_status=http.client.OK)
        self.head(collection_url, expected_status=http.client.OK)
        self.assertValidRoleAssignmentListResponse(r, resource_url=collection_url)
        existing_assignments = len(r.result.get('role_assignments'))
        gd_entity = self.build_role_assignment_entity(domain_id=self.domain_id, group_id=self.group_id, role_id=self.role_id)
        self.put(gd_entity['links']['assignment'])
        r = self.get(collection_url, expected_status=http.client.OK)
        self.head(collection_url, expected_status=http.client.OK)
        self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments + 1, resource_url=collection_url)
        self.assertRoleAssignmentInListResponse(r, gd_entity)
        collection_url = '/role_assignments?effective'
        r = self.get(collection_url, expected_status=http.client.OK)
        self.head(collection_url, expected_status=http.client.OK)
        self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments + 2, resource_url=collection_url)
        collection_url = '/role_assignments?effective=0'
        r = self.get(collection_url, expected_status=http.client.OK)
        self.head(collection_url, expected_status=http.client.OK)
        self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments + 1, resource_url=collection_url)
        collection_url = '/role_assignments?effective=False'
        r = self.get(collection_url, expected_status=http.client.OK)
        self.head(collection_url, expected_status=http.client.OK)
        self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments + 2, resource_url=collection_url)
        collection_url = '/role_assignments?effective=True'
        r = self.get(collection_url, expected_status=http.client.OK)
        self.head(collection_url, expected_status=http.client.OK)
        self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments + 2, resource_url=collection_url)

    def test_filtered_role_assignments(self):
        """Call ``GET /role_assignments?filters``.

        Test Plan:

        - Create extra users, group, role and project for tests
        - Make the following assignments:
          Give group1, role1 on project1 and domain
          Give user1, role2 on project1 and domain
          Make User1 a member of Group1
        - Test a series of single filter list calls, checking that
          the correct results are obtained
        - Test a multi-filtered list call
        - Test listing all effective roles for a given user
        - Test the equivalent of the list of roles in a project scoped
          token (all effective roles for a user on a project)

        """
        user1 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
        user2 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
        group1 = unit.new_group_ref(domain_id=self.domain['id'])
        group1 = PROVIDERS.identity_api.create_group(group1)
        PROVIDERS.identity_api.add_user_to_group(user1['id'], group1['id'])
        PROVIDERS.identity_api.add_user_to_group(user2['id'], group1['id'])
        project1 = unit.new_project_ref(domain_id=self.domain['id'])
        PROVIDERS.resource_api.create_project(project1['id'], project1)
        self.role1 = unit.new_role_ref()
        PROVIDERS.role_api.create_role(self.role1['id'], self.role1)
        self.role2 = unit.new_role_ref()
        PROVIDERS.role_api.create_role(self.role2['id'], self.role2)
        gd_entity = self.build_role_assignment_entity(domain_id=self.domain_id, group_id=group1['id'], role_id=self.role1['id'])
        self.put(gd_entity['links']['assignment'])
        ud_entity = self.build_role_assignment_entity(domain_id=self.domain_id, user_id=user1['id'], role_id=self.role2['id'])
        self.put(ud_entity['links']['assignment'])
        gp_entity = self.build_role_assignment_entity(project_id=project1['id'], group_id=group1['id'], role_id=self.role1['id'])
        self.put(gp_entity['links']['assignment'])
        up_entity = self.build_role_assignment_entity(project_id=project1['id'], user_id=user1['id'], role_id=self.role2['id'])
        self.put(up_entity['links']['assignment'])
        gs_entity = self.build_role_assignment_entity(system='all', group_id=group1['id'], role_id=self.role1['id'])
        self.put(gs_entity['links']['assignment'])
        us_entity = self.build_role_assignment_entity(system='all', user_id=user1['id'], role_id=self.role2['id'])
        self.put(us_entity['links']['assignment'])
        us2_entity = self.build_role_assignment_entity(system='all', user_id=user2['id'], role_id=self.role2['id'])
        self.put(us2_entity['links']['assignment'])
        collection_url = '/role_assignments?scope.project.id=%s' % project1['id']
        r = self.get(collection_url, expected_status=http.client.OK)
        self.head(collection_url, expected_status=http.client.OK)
        self.assertValidRoleAssignmentListResponse(r, expected_length=2, resource_url=collection_url)
        self.assertRoleAssignmentInListResponse(r, up_entity)
        self.assertRoleAssignmentInListResponse(r, gp_entity)
        collection_url = '/role_assignments?scope.domain.id=%s' % self.domain['id']
        r = self.get(collection_url, expected_status=http.client.OK)
        self.head(collection_url, expected_status=http.client.OK)
        self.assertValidRoleAssignmentListResponse(r, expected_length=2, resource_url=collection_url)
        self.assertRoleAssignmentInListResponse(r, ud_entity)
        self.assertRoleAssignmentInListResponse(r, gd_entity)
        collection_url = '/role_assignments?user.id=%s' % user1['id']
        r = self.get(collection_url, expected_status=http.client.OK)
        self.head(collection_url, expected_status=http.client.OK)
        self.assertValidRoleAssignmentListResponse(r, expected_length=3, resource_url=collection_url)
        self.assertRoleAssignmentInListResponse(r, up_entity)
        self.assertRoleAssignmentInListResponse(r, ud_entity)
        collection_url = '/role_assignments?group.id=%s' % group1['id']
        r = self.get(collection_url, expected_status=http.client.OK)
        self.head(collection_url, expected_status=http.client.OK)
        self.assertValidRoleAssignmentListResponse(r, expected_length=3, resource_url=collection_url)
        self.assertRoleAssignmentInListResponse(r, gd_entity)
        self.assertRoleAssignmentInListResponse(r, gp_entity)
        collection_url = '/role_assignments?role.id=%s' % self.role1['id']
        r = self.get(collection_url, expected_status=http.client.OK)
        self.head(collection_url, expected_status=http.client.OK)
        self.assertValidRoleAssignmentListResponse(r, expected_length=3, resource_url=collection_url)
        self.assertRoleAssignmentInListResponse(r, gd_entity)
        self.assertRoleAssignmentInListResponse(r, gp_entity)
        self.assertRoleAssignmentInListResponse(r, gs_entity)
        collection_url = '/role_assignments?role.id=%s' % self.role2['id']
        r = self.get(collection_url, expected_status=http.client.OK)
        self.head(collection_url, expected_status=http.client.OK)
        self.assertValidRoleAssignmentListResponse(r, expected_length=4, resource_url=collection_url)
        self.assertRoleAssignmentInListResponse(r, ud_entity)
        self.assertRoleAssignmentInListResponse(r, up_entity)
        self.assertRoleAssignmentInListResponse(r, us_entity)
        collection_url = '/role_assignments?user.id=%(user_id)s&scope.project.id=%(project_id)s' % {'user_id': user1['id'], 'project_id': project1['id']}
        r = self.get(collection_url, expected_status=http.client.OK)
        self.head(collection_url, expected_status=http.client.OK)
        self.assertValidRoleAssignmentListResponse(r, expected_length=1, resource_url=collection_url)
        self.assertRoleAssignmentInListResponse(r, up_entity)
        collection_url = '/role_assignments?effective&user.id=%s' % user1['id']
        r = self.get(collection_url, expected_status=http.client.OK)
        self.head(collection_url, expected_status=http.client.OK)
        self.assertValidRoleAssignmentListResponse(r, expected_length=4, resource_url=collection_url)
        self.assertRoleAssignmentInListResponse(r, up_entity)
        self.assertRoleAssignmentInListResponse(r, ud_entity)
        gp1_link = self.build_role_assignment_link(project_id=project1['id'], group_id=group1['id'], role_id=self.role1['id'])
        gd1_link = self.build_role_assignment_link(domain_id=self.domain_id, group_id=group1['id'], role_id=self.role1['id'])
        up1_entity = self.build_role_assignment_entity(link=gp1_link, project_id=project1['id'], user_id=user1['id'], role_id=self.role1['id'])
        ud1_entity = self.build_role_assignment_entity(link=gd1_link, domain_id=self.domain_id, user_id=user1['id'], role_id=self.role1['id'])
        self.assertRoleAssignmentInListResponse(r, up1_entity)
        self.assertRoleAssignmentInListResponse(r, ud1_entity)
        collection_url = '/role_assignments?effective&user.id=%(user_id)s&scope.project.id=%(project_id)s' % {'user_id': user1['id'], 'project_id': project1['id']}
        r = self.get(collection_url, expected_status=http.client.OK)
        self.head(collection_url, expected_status=http.client.OK)
        self.assertValidRoleAssignmentListResponse(r, expected_length=2, resource_url=collection_url)
        self.assertRoleAssignmentInListResponse(r, up_entity)
        self.assertRoleAssignmentInListResponse(r, up1_entity)

    def test_list_system_role_assignments(self):
        user_system_role_id = self._create_new_role()
        user_domain_role_id = self._create_new_role()
        user_project_role_id = self._create_new_role()
        group_system_role_id = self._create_new_role()
        group_domain_role_id = self._create_new_role()
        group_project_role_id = self._create_new_role()
        user = self._create_user()
        url = '/system/users/%s/roles/%s' % (user['id'], user_system_role_id)
        self.put(url)
        url = '/domains/%s/users/%s/roles/%s' % (self.domain_id, user['id'], user_domain_role_id)
        self.put(url)
        url = '/projects/%s/users/%s/roles/%s' % (self.project_id, user['id'], user_project_role_id)
        self.put(url)
        group = self._create_group()
        url = '/system/groups/%s/roles/%s' % (group['id'], group_system_role_id)
        self.put(url)
        url = '/domains/%s/groups/%s/roles/%s' % (self.domain_id, group['id'], group_domain_role_id)
        self.put(url)
        url = '/projects/%s/groups/%s/roles/%s' % (self.project_id, group['id'], group_project_role_id)
        self.put(url)
        response = self.get('/role_assignments?scope.system=all')
        self.assertValidRoleAssignmentListResponse(response, expected_length=2)
        for assignment in response.json_body['role_assignments']:
            self.assertTrue(assignment['scope']['system']['all'])
            if assignment.get('user'):
                self.assertEqual(user_system_role_id, assignment['role']['id'])
            if assignment.get('group'):
                self.assertEqual(group_system_role_id, assignment['role']['id'])
        url = '/role_assignments?scope.system=all&user.id=%s' % user['id']
        response = self.get(url)
        self.assertValidRoleAssignmentListResponse(response, expected_length=1)
        self.assertEqual(user_system_role_id, response.json_body['role_assignments'][0]['role']['id'])
        url = '/role_assignments?scope.system=all&group.id=%s' % group['id']
        response = self.get(url)
        self.assertValidRoleAssignmentListResponse(response, expected_length=1)
        self.assertEqual(group_system_role_id, response.json_body['role_assignments'][0]['role']['id'])
        url = '/role_assignments?user.id=%s' % user['id']
        response = self.get(url)
        self.assertValidRoleAssignmentListResponse(response, expected_length=3)
        for assignment in response.json_body['role_assignments']:
            if 'system' in assignment['scope']:
                self.assertEqual(user_system_role_id, assignment['role']['id'])
            if 'domain' in assignment['scope']:
                self.assertEqual(user_domain_role_id, assignment['role']['id'])
            if 'project' in assignment['scope']:
                self.assertEqual(user_project_role_id, assignment['role']['id'])
        url = '/role_assignments?group.id=%s' % group['id']
        response = self.get(url)
        self.assertValidRoleAssignmentListResponse(response, expected_length=3)
        for assignment in response.json_body['role_assignments']:
            if 'system' in assignment['scope']:
                self.assertEqual(group_system_role_id, assignment['role']['id'])
            if 'domain' in assignment['scope']:
                self.assertEqual(group_domain_role_id, assignment['role']['id'])
            if 'project' in assignment['scope']:
                self.assertEqual(group_project_role_id, assignment['role']['id'])