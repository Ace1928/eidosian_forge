import re
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
import testtools
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import firewallrule
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.fwaas import common
from neutronclient.tests.unit.osc.v2.fwaas import fakes
class TestCreateFirewallRule(TestFirewallRule, common.TestCreateFWaaS):

    def setUp(self):
        super(TestCreateFirewallRule, self).setUp()
        self.networkclient.create_firewall_rule = mock.Mock(return_value=_fwr)
        self.mocked = self.networkclient.create_firewall_rule

        def _mock_find_group(*args, **kwargs):
            return {'id': args[0]}
        self.networkclient.find_firewall_group.side_effect = _mock_find_group
        self.cmd = firewallrule.CreateFirewallRule(self.app, self.namespace)

    def _update_expect_response(self, request, response):
        """Set expected request and response

        :param request
            A dictionary of request body(dict of verifylist)
        :param response
            A OrderedDict of request body
        """
        self.networkclient.create_firewall_rule.return_value = response
        osc_utils.find_project.return_value.id = response['tenant_id']
        self.data = _generate_data(ordered_dict=response)
        self.ordered_data = tuple((_replace_display_columns(column, response[column]) for column in self.ordered_columns))

    def _set_all_params(self, args={}):
        name = args.get('name') or 'my-name'
        description = args.get('description') or 'my-desc'
        source_ip = args.get('source_ip_address') or '192.168.1.0/24'
        destination_ip = args.get('destination_ip_address') or '192.168.2.0/24'
        source_port = args.get('source_port') or '0:65535'
        protocol = args.get('protocol') or 'udp'
        action = args.get('action') or 'deny'
        ip_version = args.get('ip_version') or '4'
        destination_port = args.get('destination_port') or '0:65535'
        destination_firewall_group = args.get('destination_firewall_group') or 'my-dst-fwg'
        source_firewall_group = args.get('source_firewall_group') or 'my-src-fwg'
        tenant_id = args.get('tenant_id') or 'my-tenant'
        arglist = ['--description', description, '--name', name, '--protocol', protocol, '--ip-version', ip_version, '--source-ip-address', source_ip, '--destination-ip-address', destination_ip, '--source-port', source_port, '--destination-port', destination_port, '--action', action, '--project', tenant_id, '--disable-rule', '--share', '--source-firewall-group', source_firewall_group, '--destination-firewall-group', destination_firewall_group]
        verifylist = [('name', name), ('description', description), ('share', True), ('protocol', protocol), ('ip_version', ip_version), ('source_ip_address', source_ip), ('destination_ip_address', destination_ip), ('source_port', source_port), ('destination_port', destination_port), ('action', action), ('disable_rule', True), ('project', tenant_id), ('source_firewall_group', source_firewall_group), ('destination_firewall_group', destination_firewall_group)]
        return (arglist, verifylist)

    def _test_create_with_all_params(self, args={}):
        arglist, verifylist = self._set_all_params(args)
        request, response = _generate_req_and_res(verifylist)
        self._update_expect_response(request, response)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.check_results(headers, data, request)

    def test_create_with_no_options(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.check_results(headers, data, None)

    def test_create_with_all_params(self):
        self._test_create_with_all_params()

    def test_create_with_all_params_protocol_any(self):
        self._test_create_with_all_params({'protocol': 'any'})

    def test_create_with_all_params_ip_version_6(self):
        self._test_create_with_all_params({'ip_version': '6'})

    def test_create_with_all_params_invalid_ip_version(self):
        arglist, verifylist = self._set_all_params({'ip_version': '128'})
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_with_all_params_action_upper_capitalized(self):
        for action in ('Allow', 'DENY', 'Reject'):
            arglist, verifylist = self._set_all_params({'action': action})
            self.assertRaises(testtools.matchers._impl.MismatchError, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_with_all_params_protocol_upper_capitalized(self):
        for protocol in ('TCP', 'Tcp', 'ANY', 'AnY', 'iCMp'):
            arglist, verifylist = self._set_all_params({'protocol': protocol})
            self.assertRaises(testtools.matchers._impl.MismatchError, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_with_src_fwg_and_no(self):
        fwg = 'my-fwg'
        arglist = ['--source-firewall-group', fwg, '--no-source-firewall-group']
        verifylist = [('source_firewall_group', fwg), ('no_source_firewall_group', True)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_with_dst_fwg_and_no(self):
        fwg = 'my-fwg'
        arglist = ['--destination-firewall-group', fwg, '--no-destination-firewall-group']
        verifylist = [('destination_firewall_group', fwg), ('no_destination_firewall_group', True)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)