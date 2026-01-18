import copy
from keystoneclient import exceptions as identity_exc
from osc_lib import exceptions
from openstackclient.identity.v3 import access_rule
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestAccessRuleList(TestAccessRule):

    def setUp(self):
        super(TestAccessRuleList, self).setUp()
        self.access_rules_mock.list.return_value = [fakes.FakeResource(None, copy.deepcopy(identity_fakes.ACCESS_RULE), loaded=True)]
        self.cmd = access_rule.ListAccessRule(self.app, None)

    def test_access_rule_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.access_rules_mock.list.assert_called_with(user=None)
        collist = ('ID', 'Service', 'Method', 'Path')
        self.assertEqual(collist, columns)
        datalist = ((identity_fakes.access_rule_id, identity_fakes.access_rule_service, identity_fakes.access_rule_method, identity_fakes.access_rule_path),)
        self.assertEqual(datalist, tuple(data))