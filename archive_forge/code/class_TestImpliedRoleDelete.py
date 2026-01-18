import copy
from openstackclient.identity.v3 import implied_role
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestImpliedRoleDelete(TestRole):

    def setUp(self):
        super(TestImpliedRoleDelete, self).setUp()
        self.roles_mock.list.return_value = [fakes.FakeResource(None, copy.deepcopy(identity_fakes.ROLES[0]), loaded=True), fakes.FakeResource(None, copy.deepcopy(identity_fakes.ROLES[1]), loaded=True)]
        fake_resource = fakes.FakeResource(None, {'prior-role': copy.deepcopy(identity_fakes.ROLES[0]), 'implied': copy.deepcopy(identity_fakes.ROLES[1])}, loaded=True)
        self.inference_rules_mock.delete.return_value = fake_resource
        self.cmd = implied_role.DeleteImpliedRole(self.app, None)

    def test_implied_role_delete(self):
        arglist = [identity_fakes.ROLES[0]['id'], '--implied-role', identity_fakes.ROLES[1]['id']]
        verifylist = [('role', identity_fakes.ROLES[0]['id']), ('implied_role', identity_fakes.ROLES[1]['id'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.inference_rules_mock.delete.assert_called_with(identity_fakes.ROLES[0]['id'], identity_fakes.ROLES[1]['id'])