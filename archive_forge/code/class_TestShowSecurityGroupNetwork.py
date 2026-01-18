from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import security_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestShowSecurityGroupNetwork(TestSecurityGroupNetwork):
    _security_group_rule = network_fakes.FakeSecurityGroupRule.create_one_security_group_rule()
    _security_group = network_fakes.FakeSecurityGroup.create_one_security_group(attrs={'security_group_rules': [_security_group_rule._info]})
    columns = ('description', 'id', 'name', 'project_id', 'rules', 'stateful', 'tags')
    data = (_security_group.description, _security_group.id, _security_group.name, _security_group.project_id, security_group.NetworkSecurityGroupRulesColumn([_security_group_rule._info]), _security_group.stateful, _security_group.tags)

    def setUp(self):
        super(TestShowSecurityGroupNetwork, self).setUp()
        self.network_client.find_security_group = mock.Mock(return_value=self._security_group)
        self.cmd = security_group.ShowSecurityGroup(self.app, self.namespace)

    def test_show_no_options(self):
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, [], [])

    def test_show_all_options(self):
        arglist = [self._security_group.id]
        verifylist = [('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_security_group.assert_called_once_with(self._security_group.id, ignore_missing=False)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)