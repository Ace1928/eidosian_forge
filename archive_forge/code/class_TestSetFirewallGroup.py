import copy
import re
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import firewallgroup
from neutronclient.osc.v2 import utils as v2_utils
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.fwaas import common
from neutronclient.tests.unit.osc.v2.fwaas import fakes
class TestSetFirewallGroup(TestFirewallGroup, common.TestSetFWaaS):

    def setUp(self):
        super(TestSetFirewallGroup, self).setUp()
        _fwg['ports'] = ['old_port']
        self.networkclient.update_firewall_group = mock.Mock(return_value={self.res: _fwg})
        self.mocked = self.networkclient.update_firewall_group

        def _mock_find_port(*args, **kwargs):
            return {'id': args[0]}
        self.networkclient.find_port.side_effect = _mock_find_port
        self.cmd = firewallgroup.SetFirewallGroup(self.app, self.namespace)

    def _update_expect_response(self, request, response):
        """Set expected request and response

        :param request
            A dictionary of request body(dict of verifylist)
        :param response
            A OrderedDict of request body
        """
        osc_utils.find_project.return_value.id = response['tenant_id']
        self.data = _generate_response(ordered_dict=response)
        self.ordered_data = tuple((response[column] for column in self.ordered_columns))

    def test_set_ingress_policy_and_egress_policy(self):
        target = self.resource['id']
        ingress_policy = 'ingress_policy'
        egress_policy = 'egress_policy'

        def _mock_fwg_policy(*args, **kwargs):
            if self.networkclient.find_firewall_group.call_count == 1:
                self.networkclient.find_firewall_group.assert_called_with(target)
            if self.networkclient.find_firewall_policy.call_count == 1:
                self.networkclient.find_firewall_policy.assert_called_with(ingress_policy)
            if self.networkclient.find_firewall_policy.call_count == 2:
                self.networkclient.find_firewall_policy.assert_called_with(egress_policy)
            return {'id': args[0]}
        self.networkclient.find_firewall_group.side_effect = _mock_fwg_policy
        self.networkclient.find_firewall_policy.side_effect = _mock_fwg_policy
        arglist = [target, '--ingress-firewall-policy', ingress_policy, '--egress-firewall-policy', egress_policy]
        verifylist = [(self.res, target), ('ingress_firewall_policy', ingress_policy), ('egress_firewall_policy', egress_policy)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'ingress_firewall_policy_id': ingress_policy, 'egress_firewall_policy_id': egress_policy})
        self.assertIsNone(result)

    def test_set_port(self):
        target = self.resource['id']
        port1 = 'additional_port1'
        port2 = 'additional_port2'

        def _mock_port_fwg(*args, **kwargs):
            if self.networkclient.find_firewall_group.call_count in [1, 2]:
                self.networkclient.find_firewall_group.assert_called_with(target)
                return {'id': args[0], 'ports': _fwg['ports']}
            if self.networkclient.find_port.call_count == 1:
                self.networkclient.find_port.assert_called_with(args)
                return {'id': args[0]}
            if self.networkclient.find_port.call_count == 2:
                self.networkclient.find_port.assert_called_with(args)
                return {'id': args[0]}
        self.networkclient.find_fireall_group.side_effect = _mock_port_fwg
        self.networkclient.find_port.side_effect = _mock_port_fwg
        arglist = [target, '--port', port1, '--port', port2]
        verifylist = [(self.res, target), ('port', [port1, port2])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        expect = {'ports': sorted(_fwg['ports'] + [port1, port2])}
        self.mocked.assert_called_once_with(target, **expect)
        self.assertEqual(2, self.networkclient.find_firewall_group.call_count)
        self.assertIsNone(result)

    def test_set_no_port(self):
        target = self.resource['id']
        arglist = [target, '--no-port']
        verifylist = [(self.res, target), ('no_port', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'ports': []})
        self.assertIsNone(result)

    def test_set_admin_state(self):
        target = self.resource['id']
        arglist = [target, '--enable']
        verifylist = [(self.res, target), ('enable', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'admin_state_up': True})
        self.assertIsNone(result)

    def test_set_egress_policy(self):
        target = self.resource['id']
        policy = 'egress_policy'

        def _mock_find_policy(*args, **kwargs):
            return {'id': args[0]}
        self.networkclient.find_firewall_policy.side_effect = _mock_find_policy
        arglist = [target, '--egress-firewall-policy', policy]
        verifylist = [(self.res, target), ('egress_firewall_policy', policy)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'egress_firewall_policy_id': policy})
        self.assertIsNone(result)

    def test_set_no_ingress_policies(self):
        target = self.resource['id']
        arglist = [target, '--no-ingress-firewall-policy']
        verifylist = [(self.res, target), ('no_ingress_firewall_policy', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'ingress_firewall_policy_id': None})
        self.assertIsNone(result)

    def test_set_no_egress_policies(self):
        target = self.resource['id']
        arglist = [target, '--no-egress-firewall-policy']
        verifylist = [(self.res, target), ('no_egress_firewall_policy', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'egress_firewall_policy_id': None})
        self.assertIsNone(result)

    def test_set_port_and_no_port(self):
        target = self.resource['id']
        port = 'my-port'
        arglist = [target, '--port', port, '--no-port']
        verifylist = [(self.res, target), ('port', [port]), ('no_port', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'ports': [port]})
        self.assertIsNone(result)

    def test_set_ingress_policy_and_no_ingress_policy(self):
        target = self.resource['id']
        arglist = [target, '--ingress-firewall-policy', 'my-ingress', '--no-ingress-firewall-policy']
        verifylist = [(self.res, target), ('ingress_firewall_policy', 'my-ingress'), ('no_ingress_firewall_policy', True)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_set_egress_policy_and_no_egress_policy(self):
        target = self.resource['id']
        arglist = [target, '--egress-firewall-policy', 'my-egress', '--no-egress-firewall-policy']
        verifylist = [(self.res, target), ('egress_firewall_policy', 'my-egress'), ('no_egress_firewall_policy', True)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_set_and_raises(self):
        self.networkclient.update_firewall_group = mock.Mock(side_effect=Exception)
        target = self.resource['id']
        arglist = [target, '--name', 'my-name']
        verifylist = [(self.res, target), ('name', 'my-name')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)