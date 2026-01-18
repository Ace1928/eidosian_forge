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
class _SystemUserTests(object):
    """Common functionality for system users regardless of default role."""

    def test_user_can_list_all_role_assignments_in_the_deployment(self):
        assignments = self._setup_test_role_assignments()
        self.expected.append({'user_id': self.bootstrapper.admin_user_id, 'project_id': self.bootstrapper.project_id, 'role_id': self.bootstrapper.admin_role_id})
        self.expected.append({'user_id': self.bootstrapper.admin_user_id, 'system': 'all', 'role_id': self.bootstrapper.admin_role_id})
        self.expected.append({'user_id': assignments['user_id'], 'project_id': assignments['project_id'], 'role_id': assignments['role_id']})
        self.expected.append({'user_id': assignments['user_id'], 'domain_id': assignments['domain_id'], 'role_id': assignments['role_id']})
        self.expected.append({'user_id': assignments['user_id'], 'system': 'all', 'role_id': assignments['role_id']})
        self.expected.append({'group_id': assignments['group_id'], 'project_id': assignments['project_id'], 'role_id': assignments['role_id']})
        self.expected.append({'group_id': assignments['group_id'], 'domain_id': assignments['domain_id'], 'role_id': assignments['role_id']})
        self.expected.append({'group_id': assignments['group_id'], 'system': 'all', 'role_id': assignments['role_id']})
        with self.test_client() as c:
            r = c.get('/v3/role_assignments', headers=self.headers)
            self.assertEqual(len(self.expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, self.expected)

    def test_user_can_list_all_role_names_assignments_in_the_deployment(self):
        assignments = self._setup_test_role_assignments()
        self.expected.append({'user_id': self.bootstrapper.admin_user_id, 'project_id': self.bootstrapper.project_id, 'role_id': self.bootstrapper.admin_role_id})
        self.expected.append({'user_id': self.bootstrapper.admin_user_id, 'system': 'all', 'role_id': self.bootstrapper.admin_role_id})
        self.expected.append({'user_id': assignments['user_id'], 'project_id': assignments['project_id'], 'role_id': assignments['role_id']})
        self.expected.append({'user_id': assignments['user_id'], 'domain_id': assignments['domain_id'], 'role_id': assignments['role_id']})
        self.expected.append({'user_id': assignments['user_id'], 'system': 'all', 'role_id': assignments['role_id']})
        self.expected.append({'group_id': assignments['group_id'], 'project_id': assignments['project_id'], 'role_id': assignments['role_id']})
        self.expected.append({'group_id': assignments['group_id'], 'domain_id': assignments['domain_id'], 'role_id': assignments['role_id']})
        self.expected.append({'group_id': assignments['group_id'], 'system': 'all', 'role_id': assignments['role_id']})
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?include_names=True', headers=self.headers)
            self.assertEqual(len(self.expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, self.expected)

    def test_user_can_filter_role_assignments_by_project(self):
        assignments = self._setup_test_role_assignments()
        expected = [{'user_id': assignments['user_id'], 'project_id': assignments['project_id'], 'role_id': assignments['role_id']}, {'group_id': assignments['group_id'], 'project_id': assignments['project_id'], 'role_id': assignments['role_id']}]
        project_id = assignments['project_id']
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?scope.project.id=%s' % project_id, headers=self.headers)
            self.assertEqual(len(expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, expected)

    def test_user_can_filter_role_assignments_by_domain(self):
        assignments = self._setup_test_role_assignments()
        expected = [{'user_id': assignments['user_id'], 'domain_id': assignments['domain_id'], 'role_id': assignments['role_id']}, {'group_id': assignments['group_id'], 'domain_id': assignments['domain_id'], 'role_id': assignments['role_id']}]
        domain_id = assignments['domain_id']
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?scope.domain.id=%s' % domain_id, headers=self.headers)
            self.assertEqual(len(expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, expected)

    def test_user_can_filter_role_assignments_by_system(self):
        assignments = self._setup_test_role_assignments()
        self.expected.append({'user_id': self.bootstrapper.admin_user_id, 'system': 'all', 'role_id': self.bootstrapper.admin_role_id})
        self.expected.append({'user_id': assignments['user_id'], 'system': 'all', 'role_id': assignments['role_id']})
        self.expected.append({'group_id': assignments['group_id'], 'system': 'all', 'role_id': assignments['role_id']})
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?scope.system=all', headers=self.headers)
            self.assertEqual(len(self.expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, self.expected)

    def test_user_can_filter_role_assignments_by_user(self):
        assignments = self._setup_test_role_assignments()
        expected = [{'user_id': assignments['user_id'], 'project_id': assignments['project_id'], 'role_id': assignments['role_id']}, {'user_id': assignments['user_id'], 'domain_id': assignments['domain_id'], 'role_id': assignments['role_id']}, {'user_id': assignments['user_id'], 'system': 'all', 'role_id': assignments['role_id']}]
        user_id = assignments['user_id']
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?user.id=%s' % user_id, headers=self.headers)
            self.assertEqual(len(expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, expected)

    def test_user_can_filter_role_assignments_by_group(self):
        assignments = self._setup_test_role_assignments()
        expected = [{'group_id': assignments['group_id'], 'project_id': assignments['project_id'], 'role_id': assignments['role_id']}, {'group_id': assignments['group_id'], 'domain_id': assignments['domain_id'], 'role_id': assignments['role_id']}, {'group_id': assignments['group_id'], 'system': 'all', 'role_id': assignments['role_id']}]
        group_id = assignments['group_id']
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?group.id=%s' % group_id, headers=self.headers)
            self.assertEqual(len(expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, expected)

    def test_user_can_filter_role_assignments_by_role(self):
        assignments = self._setup_test_role_assignments()
        self.expected = [ra for ra in self.expected if ra['role_id'] == assignments['role_id']]
        self.expected.append({'user_id': assignments['user_id'], 'project_id': assignments['project_id'], 'role_id': assignments['role_id']})
        self.expected.append({'user_id': assignments['user_id'], 'domain_id': assignments['domain_id'], 'role_id': assignments['role_id']})
        self.expected.append({'user_id': assignments['user_id'], 'system': 'all', 'role_id': assignments['role_id']})
        self.expected.append({'group_id': assignments['group_id'], 'project_id': assignments['project_id'], 'role_id': assignments['role_id']})
        self.expected.append({'group_id': assignments['group_id'], 'domain_id': assignments['domain_id'], 'role_id': assignments['role_id']})
        self.expected.append({'group_id': assignments['group_id'], 'system': 'all', 'role_id': assignments['role_id']})
        role_id = assignments['role_id']
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?role.id=%s&include_names=True' % role_id, headers=self.headers)
            self.assertEqual(len(self.expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, self.expected)

    def test_user_can_filter_role_assignments_by_project_and_role(self):
        assignments = self._setup_test_role_assignments()
        expected = [{'user_id': assignments['user_id'], 'project_id': assignments['project_id'], 'role_id': assignments['role_id']}, {'group_id': assignments['group_id'], 'project_id': assignments['project_id'], 'role_id': assignments['role_id']}]
        with self.test_client() as c:
            qs = (assignments['project_id'], assignments['role_id'])
            r = c.get('/v3/role_assignments?scope.project.id=%s&role.id=%s' % qs, headers=self.headers)
            self.assertEqual(len(expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, expected)

    def test_user_can_filter_role_assignments_by_domain_and_role(self):
        assignments = self._setup_test_role_assignments()
        expected = [{'user_id': assignments['user_id'], 'domain_id': assignments['domain_id'], 'role_id': assignments['role_id']}, {'group_id': assignments['group_id'], 'domain_id': assignments['domain_id'], 'role_id': assignments['role_id']}]
        qs = (assignments['domain_id'], assignments['role_id'])
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?scope.domain.id=%s&role.id=%s' % qs, headers=self.headers)
            self.assertEqual(len(expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, expected)

    def test_user_can_filter_role_assignments_by_system_and_role(self):
        assignments = self._setup_test_role_assignments()
        self.expected = [ra for ra in self.expected if ra['role_id'] == assignments['role_id']]
        self.expected.append({'user_id': assignments['user_id'], 'system': 'all', 'role_id': assignments['role_id']})
        self.expected.append({'group_id': assignments['group_id'], 'system': 'all', 'role_id': assignments['role_id']})
        role_id = assignments['role_id']
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?scope.system=all&role.id=%s' % role_id, headers=self.headers)
            self.assertEqual(len(self.expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, self.expected)

    def test_user_can_filter_role_assignments_by_user_and_role(self):
        assignments = self._setup_test_role_assignments()
        expected = [{'user_id': assignments['user_id'], 'project_id': assignments['project_id'], 'role_id': assignments['role_id']}, {'user_id': assignments['user_id'], 'domain_id': assignments['domain_id'], 'role_id': assignments['role_id']}, {'user_id': assignments['user_id'], 'system': 'all', 'role_id': assignments['role_id']}]
        qs = (assignments['user_id'], assignments['role_id'])
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?user.id=%s&role.id=%s' % qs, headers=self.headers)
            self.assertEqual(len(expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, expected)

    def test_user_can_filter_role_assignments_by_group_and_role(self):
        assignments = self._setup_test_role_assignments()
        expected = [{'group_id': assignments['group_id'], 'project_id': assignments['project_id'], 'role_id': assignments['role_id']}, {'group_id': assignments['group_id'], 'domain_id': assignments['domain_id'], 'role_id': assignments['role_id']}, {'group_id': assignments['group_id'], 'system': 'all', 'role_id': assignments['role_id']}]
        with self.test_client() as c:
            qs = (assignments['group_id'], assignments['role_id'])
            r = c.get('/v3/role_assignments?group.id=%s&role.id=%s' % qs, headers=self.headers)
            self.assertEqual(len(expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, expected)

    def test_user_can_filter_role_assignments_by_project_and_user(self):
        assignments = self._setup_test_role_assignments()
        expected = [{'user_id': assignments['user_id'], 'project_id': assignments['project_id'], 'role_id': assignments['role_id']}]
        qs = (assignments['project_id'], assignments['user_id'])
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?scope.project.id=%s&user.id=%s' % qs, headers=self.headers)
            self.assertEqual(len(expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, expected)

    def test_user_can_filter_role_assignments_by_project_and_group(self):
        assignments = self._setup_test_role_assignments()
        expected = [{'group_id': assignments['group_id'], 'project_id': assignments['project_id'], 'role_id': assignments['role_id']}]
        qs = (assignments['project_id'], assignments['group_id'])
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?scope.project.id=%s&group.id=%s' % qs, headers=self.headers)
            self.assertEqual(len(expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, expected)

    def test_user_can_filter_role_assignments_by_domain_and_user(self):
        assignments = self._setup_test_role_assignments()
        expected = [{'user_id': assignments['user_id'], 'domain_id': assignments['domain_id'], 'role_id': assignments['role_id']}]
        qs = (assignments['domain_id'], assignments['user_id'])
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?scope.domain.id=%s&user.id=%s' % qs, headers=self.headers)
            self.assertEqual(len(expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, expected)

    def test_user_can_filter_role_assignments_by_domain_and_group(self):
        assignments = self._setup_test_role_assignments()
        expected = [{'group_id': assignments['group_id'], 'domain_id': assignments['domain_id'], 'role_id': assignments['role_id']}]
        qs = (assignments['domain_id'], assignments['group_id'])
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?scope.domain.id=%s&group.id=%s' % qs, headers=self.headers)
            self.assertEqual(len(expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, expected)

    def test_user_can_list_assignments_for_subtree(self):
        assignments = self._setup_test_role_assignments()
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=CONF.identity.default_domain_id))
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id, parent_id=assignments['project_id']))
        PROVIDERS.assignment_api.create_grant(assignments['role_id'], user_id=user['id'], project_id=project['id'])
        expected = [{'user_id': assignments['user_id'], 'project_id': assignments['project_id'], 'role_id': assignments['role_id']}, {'group_id': assignments['group_id'], 'project_id': assignments['project_id'], 'role_id': assignments['role_id']}, {'user_id': user['id'], 'project_id': project['id'], 'role_id': assignments['role_id']}]
        with self.test_client() as c:
            r = c.get('/v3/role_assignments?scope.project.id=%s&include_subtree' % assignments['project_id'], headers=self.headers)
            self.assertEqual(len(expected), len(r.json['role_assignments']))
            actual = self._extract_role_assignments_from_response_body(r)
            for assignment in actual:
                self.assertIn(assignment, expected)