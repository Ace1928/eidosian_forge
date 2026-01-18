import copy
from keystoneclient import exceptions as identity_exc
from osc_lib import exceptions
from openstackclient.identity.v3 import access_rule
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestAccessRuleDelete(TestAccessRule):

    def setUp(self):
        super(TestAccessRuleDelete, self).setUp()
        self.access_rules_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.ACCESS_RULE), loaded=True)
        self.access_rules_mock.delete.return_value = None
        self.cmd = access_rule.DeleteAccessRule(self.app, None)

    def test_access_rule_delete(self):
        arglist = [identity_fakes.access_rule_id]
        verifylist = [('access_rule', [identity_fakes.access_rule_id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.access_rules_mock.delete.assert_called_with(identity_fakes.access_rule_id)
        self.assertIsNone(result)

    def test_delete_multi_access_rules_with_exception(self):
        mock_get = self.access_rules_mock.get
        mock_get.side_effect = [mock_get.return_value, identity_exc.NotFound]
        arglist = [identity_fakes.access_rule_id, 'nonexistent_access_rule']
        verifylist = [('access_rule', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 access rules failed to delete.', str(e))
        mock_get.assert_any_call(identity_fakes.access_rule_id)
        mock_get.assert_any_call('nonexistent_access_rule')
        self.assertEqual(2, mock_get.call_count)
        self.access_rules_mock.delete.assert_called_once_with(identity_fakes.access_rule_id)