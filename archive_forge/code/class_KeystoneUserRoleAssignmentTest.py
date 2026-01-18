import copy
from unittest import mock
from heat.common import exception
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.keystone import role_assignments
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import generic_resource
from heat.tests import utils
class KeystoneUserRoleAssignmentTest(common.HeatTestCase):
    role_assignment_template = copy.deepcopy(keystone_role_assignment_template)
    role = role_assignment_template['resources']['test_role_assignment']
    role['properties']['user'] = 'user_1'
    role['type'] = 'OS::Keystone::UserRoleAssignment'

    def setUp(self):
        super(KeystoneUserRoleAssignmentTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.stack = stack.Stack(self.ctx, 'test_stack_keystone_user_role_add', template.Template(self.role_assignment_template))
        self.test_role_assignment = self.stack['test_role_assignment']
        self.keystoneclient = mock.Mock()
        self.patchobject(resource.Resource, 'client', return_value=fake_ks.FakeKeystoneClient(client=self.keystoneclient))
        self.roles = self.keystoneclient.roles

        def _side_effect(value):
            return value
        self.keystone_client_plugin = mock.MagicMock()
        self.keystone_client_plugin.get_user_id.side_effect = _side_effect
        self.keystone_client_plugin.get_domain_id.side_effect = _side_effect
        self.keystone_client_plugin.get_role_id.side_effect = _side_effect
        self.keystone_client_plugin.get_project_id.side_effect = _side_effect
        self.test_role_assignment.client_plugin = mock.MagicMock()
        self.test_role_assignment.client_plugin.return_value = self.keystone_client_plugin
        self.test_role_assignment.parse_list_assignments = mock.MagicMock()
        self.test_role_assignment.parse_list_assignments.return_value = [{'role': 'role_1', 'domain': 'domain_1', 'project': None}, {'role': 'role_1', 'project': 'project_1', 'domain': None}]

    def test_user_role_assignment_handle_create(self):
        self.test_role_assignment.handle_create()
        self.roles.grant.assert_any_call(role='role_1', user='user_1', domain='domain_1')
        self.roles.grant.assert_any_call(role='role_1', user='user_1', project='project_1')

    def test_user_role_assignment_handle_update(self):
        prop_diff = {MixinClass.ROLES: [{'role': 'role_2', 'project': 'project_1'}, {'role': 'role_2', 'domain': 'domain_1'}]}
        self.test_role_assignment.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
        self.roles.grant.assert_any_call(role='role_2', user='user_1', domain='domain_1')
        self.roles.grant.assert_any_call(role='role_2', user='user_1', project='project_1')
        self.roles.revoke.assert_any_call(role='role_1', user='user_1', domain='domain_1')
        self.roles.revoke.assert_any_call(role='role_1', user='user_1', project='project_1')

    def test_user_role_assignment_handle_delete(self):
        self.assertIsNone(self.test_role_assignment.handle_delete())
        self.roles.revoke.assert_any_call(role='role_1', user='user_1', domain='domain_1')
        self.roles.revoke.assert_any_call(role='role_1', user='user_1', project='project_1')

    def test_user_role_assignment_delete_user_not_found(self):
        self.keystone_client_plugin.get_user_id.side_effect = [exception.EntityNotFound]
        self.assertIsNone(self.test_role_assignment.handle_delete())
        self.roles.revoke.assert_not_called()