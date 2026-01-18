import copy
from keystoneclient import exceptions as identity_exc
from osc_lib import exceptions
from openstackclient.identity.v3 import access_rule
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestAccessRuleShow(TestAccessRule):

    def setUp(self):
        super(TestAccessRuleShow, self).setUp()
        self.access_rules_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.ACCESS_RULE), loaded=True)
        self.cmd = access_rule.ShowAccessRule(self.app, None)

    def test_access_rule_show(self):
        arglist = [identity_fakes.access_rule_id]
        verifylist = [('access_rule', identity_fakes.access_rule_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.access_rules_mock.get.assert_called_with(identity_fakes.access_rule_id)
        collist = ('id', 'method', 'path', 'service')
        self.assertEqual(collist, columns)
        datalist = (identity_fakes.access_rule_id, identity_fakes.access_rule_method, identity_fakes.access_rule_path, identity_fakes.access_rule_service)
        self.assertEqual(datalist, data)