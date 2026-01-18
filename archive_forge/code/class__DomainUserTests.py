import copy
import http.client
import uuid
from oslo_serialization import jsonutils
from keystone.common.policies import role_assignment as rp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class _DomainUserTests(object):
    """Common functionality for domain users."""

    def _setup_test_role_assignments_for_domain(self):
        role_id = self.bootstrapper.reader_role_id
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=CONF.identity.default_domain_id))
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=CONF.identity.default_domain_id))
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=self.domain_id))
        PROVIDERS.assignment_api.create_grant(role_id, user_id=user['id'], project_id=project['id'])
        PROVIDERS.assignment_api.create_grant(role_id, user_id=user['id'], domain_id=self.domain_id)
        PROVIDERS.assignment_api.create_grant(role_id, group_id=group['id'], project_id=project['id'])
        PROVIDERS.assignment_api.create_grant(role_id, group_id=group['id'], domain_id=self.domain_id)
        return {'user_id': user['id'], 'group_id': group['id'], 'project_id': project['id'], 'role_id': role_id}

    def test_user_can_list_all_assignments_in_their_domain(self):
        self._setup_test_role_assignments()
        domain_assignments = self._setup_test_role_assignments_for_domain()
        self.expected.append({'user_id': domain_assignments['user_id'], 'domain_id': self.domain_id, 'role_id': domain_assignments['role_id']})
        self.expected.append({'user_id': domain_assignments['user_id'], 'project_id': domain_assignments['project_id'], 'role_id': domain_assignments['role_id']})
        self.expected.append({'group_id': domain_assignments['group_id'], 'domain_id': self.domain_id, 'role_id': domain_assignments['role_id']})
        self.expected.append({'group_id': domain_assignments['group_id'], 'project_id': domain_assignments['project_id'], 'role_id': domain_assignments['role_id']})
        with self.test_client() as c:
            r = c.get('/v3/role_assignments', headers=self.headers)
            self.assertEqual(len(self.expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, self.expected)

    def test_user_can_filter_role_assignments_by_project_in_domain(self):
        self._setup_test_role_assignments()
        domain_assignments = self._setup_test_role_assignments_for_domain()
        expected = [{'user_id': domain_assignments['user_id'], 'project_id': domain_assignments['project_id'], 'role_id': domain_assignments['role_id']}, {'group_id': domain_assignments['group_id'], 'project_id': domain_assignments['project_id'], 'role_id': domain_assignments['role_id']}]
        project_id = domain_assignments['project_id']
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?scope.project.id=%s' % project_id, headers=self.headers)
            self.assertEqual(len(expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, expected)

    def test_user_can_filter_role_assignments_by_domain(self):
        self._setup_test_role_assignments()
        domain_assignments = self._setup_test_role_assignments_for_domain()
        self.expected.append({'user_id': domain_assignments['user_id'], 'domain_id': self.domain_id, 'role_id': domain_assignments['role_id']})
        self.expected.append({'group_id': domain_assignments['group_id'], 'domain_id': self.domain_id, 'role_id': domain_assignments['role_id']})
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?scope.domain.id=%s' % self.domain_id, headers=self.headers)
            self.assertEqual(len(self.expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, self.expected)

    def test_user_can_filter_role_assignments_by_user_of_domain(self):
        self._setup_test_role_assignments()
        domain_assignments = self._setup_test_role_assignments_for_domain()
        expected = [{'user_id': domain_assignments['user_id'], 'domain_id': self.domain_id, 'role_id': domain_assignments['role_id']}, {'user_id': domain_assignments['user_id'], 'project_id': domain_assignments['project_id'], 'role_id': domain_assignments['role_id']}]
        user_id = domain_assignments['user_id']
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?user.id=%s' % user_id, headers=self.headers)
            self.assertEqual(len(expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, expected)

    def test_user_can_filter_role_assignments_by_group_of_domain(self):
        self._setup_test_role_assignments()
        domain_assignments = self._setup_test_role_assignments_for_domain()
        expected = [{'group_id': domain_assignments['group_id'], 'domain_id': self.domain_id, 'role_id': domain_assignments['role_id']}, {'group_id': domain_assignments['group_id'], 'project_id': domain_assignments['project_id'], 'role_id': domain_assignments['role_id']}]
        group_id = domain_assignments['group_id']
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?group.id=%s' % group_id, headers=self.headers)
            self.assertEqual(len(expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, expected)

    def test_user_cannot_filter_role_assignments_by_system(self):
        self._setup_test_role_assignments()
        self._setup_test_role_assignments_for_domain()
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?scope.system=all', headers=self.headers)
            self.assertEqual(0, len(r.json['role_assignments']))

    def test_user_cannot_filter_role_assignments_by_other_domain(self):
        assignments = self._setup_test_role_assignments()
        domain = assignments['domain_id']
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?scope.domain.id=%s' % domain, headers=self.headers)
            self.assertEqual([], r.json['role_assignments'])

    def test_user_cannot_filter_role_assignments_by_other_domain_project(self):
        assignments = self._setup_test_role_assignments()
        self._setup_test_role_assignments_for_domain()
        project_id = assignments['project_id']
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?scope.project.id=%s' % project_id, headers=self.headers)
            self.assertEqual(0, len(r.json['role_assignments']))

    def test_user_cannot_filter_role_assignments_by_other_domain_user(self):
        assignments = self._setup_test_role_assignments()
        self._setup_test_role_assignments_for_domain()
        user_id = assignments['user_id']
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?user.id=%s' % user_id, headers=self.headers)
            self.assertEqual(0, len(r.json['role_assignments']))

    def test_user_cannot_filter_role_assignments_by_other_domain_group(self):
        assignments = self._setup_test_role_assignments()
        self._setup_test_role_assignments_for_domain()
        group_id = assignments['group_id']
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?group.id=%s' % group_id, headers=self.headers)
            self.assertEqual(0, len(r.json['role_assignments']))

    def test_user_can_list_assignments_for_subtree_in_their_domain(self):
        assignments = self._setup_test_role_assignments()
        domain_assignments = self._setup_test_role_assignments_for_domain()
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=self.domain_id))
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=self.domain_id, parent_id=domain_assignments['project_id']))
        PROVIDERS.assignment_api.create_grant(assignments['role_id'], user_id=user['id'], project_id=project['id'])
        expected = [{'user_id': domain_assignments['user_id'], 'project_id': domain_assignments['project_id'], 'role_id': assignments['role_id']}, {'group_id': domain_assignments['group_id'], 'project_id': domain_assignments['project_id'], 'role_id': assignments['role_id']}, {'user_id': user['id'], 'project_id': project['id'], 'role_id': assignments['role_id']}]
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?scope.project.id=%s&include_subtree' % domain_assignments['project_id'], headers=self.headers)
            self.assertEqual(len(expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, expected)

    def test_user_cannot_list_assignments_for_subtree_in_other_domain(self):
        assignments = self._setup_test_role_assignments()
        with self.test_client() as c:
            c.get('/v3/role_assignments?scope.project.id=%s&include_subtree' % assignments['project_id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)