import re
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import firewallpolicy
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.fwaas import common
from neutronclient.tests.unit.osc.v2.fwaas import fakes
class TestCreateFirewallPolicy(TestFirewallPolicy, common.TestCreateFWaaS):

    def setUp(self):
        super(TestCreateFirewallPolicy, self).setUp()
        self.networkclient.create_firewall_policy = mock.Mock(return_value={self.res: _fwp})
        self.mocked = self.networkclient.create_firewall_policy
        self.cmd = firewallpolicy.CreateFirewallPolicy(self.app, self.namespace)

    def _update_expect_response(self, request, response):
        """Set expected request and response

        :param request
            A dictionary of request body(dict of verifylist)
        :param response
            A OrderedDict of request body
        """
        self.networkclient.create_firewall_policy.return_value = response
        osc_utils.find_project.return_value.id = response['tenant_id']
        self.data = _generate_data(data=response)
        self.ordered_data = tuple((response[column] for column in self.ordered_columns))

    def test_create_with_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_with_mandatory_param(self):
        name = 'my-fwg'
        arglist = [name]
        verifylist = [('name', name)]
        request, response = _generate_req_and_res(verifylist)
        self._update_expect_response(request, response)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.check_results(headers, data, request)

    def test_create_with_rules(self):
        name = 'my-fwg'
        rule1 = 'rule1'
        rule2 = 'rule2'

        def _mock_policy(*args, **kwargs):
            return {'id': args[0]}
        self.networkclient.find_firewall_rule.side_effect = _mock_policy
        arglist = [name, '--firewall-rule', rule1, '--firewall-rule', rule2]
        verifylist = [('name', name), ('firewall_rule', [rule1, rule2])]
        request, response = _generate_req_and_res(verifylist)
        self._update_expect_response(request, response)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.assertEqual(2, self.networkclient.find_firewall_rule.call_count)
        self.check_results(headers, data, request)

    def test_create_with_all_params(self):
        name = 'my-fwp'
        desc = 'my-desc'
        rule1 = 'rule1'
        rule2 = 'rule2'
        project = 'my-tenant'

        def _mock_find(*args, **kwargs):
            if self.res in args[0]:
                rules = _fwp['firewall_rules']
                return {'id': args[0], 'firewall_rules': rules}
            return {'id': args[0]}
        self.networkclient.find_firewall_policy.side_effect = _mock_find
        self.networkclient.find_firewall_rule.side_effect = _mock_find
        arglist = [name, '--description', desc, '--firewall-rule', rule1, '--firewall-rule', rule2, '--project', project, '--share', '--audited']
        verifylist = [('name', name), ('description', desc), ('firewall_rule', [rule1, rule2]), ('project', project), ('share', True), ('audited', True)]
        request, response = _generate_req_and_res(verifylist)
        self._update_expect_response(request, response)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.check_results(headers, data, request)

    def test_create_with_firewall_rule_and_no(self):
        name = 'my-fwp'
        rule1 = 'rule1'
        rule2 = 'rule2'
        arglist = [name, '--firewall-rule', rule1, '--firewall-rule', rule2, '--no-firewall-rule']
        verifylist = [('name', name), ('firewall_rule', [rule1, rule2]), ('no_firewall_rule', True)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_with_shared_and_no_share(self):
        name = 'my-fwp'
        arglist = [name, '--share', '--no-share']
        verifylist = [('name', name), ('share', True), ('no_share', True)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_with_audited_and_no(self):
        name = 'my-fwp'
        arglist = [name, '--audited', '--no-audited']
        verifylist = [('name', name), ('audited', True), ('no_audited', True)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)