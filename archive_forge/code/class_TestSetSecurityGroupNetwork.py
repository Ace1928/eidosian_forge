from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import security_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestSetSecurityGroupNetwork(TestSecurityGroupNetwork):
    _security_group = network_fakes.FakeSecurityGroup.create_one_security_group(attrs={'tags': ['green', 'red']})

    def setUp(self):
        super(TestSetSecurityGroupNetwork, self).setUp()
        self.network_client.update_security_group = mock.Mock(return_value=None)
        self.network_client.find_security_group = mock.Mock(return_value=self._security_group)
        self.network_client.set_tags = mock.Mock(return_value=None)
        self.cmd = security_group.SetSecurityGroup(self.app, self.namespace)

    def test_set_no_options(self):
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, [], [])

    def test_set_no_updates(self):
        arglist = [self._security_group.name]
        verifylist = [('group', self._security_group.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.update_security_group.assert_called_once_with(self._security_group, **{})
        self.assertIsNone(result)

    def test_set_all_options(self):
        new_name = 'new-' + self._security_group.name
        new_description = 'new-' + self._security_group.description
        arglist = ['--name', new_name, '--description', new_description, '--stateful', self._security_group.name]
        verifylist = [('description', new_description), ('group', self._security_group.name), ('stateful', self._security_group.stateful), ('name', new_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'description': new_description, 'name': new_name, 'stateful': True}
        self.network_client.update_security_group.assert_called_once_with(self._security_group, **attrs)
        self.assertIsNone(result)

    def _test_set_tags(self, with_tags=True):
        if with_tags:
            arglist = ['--tag', 'red', '--tag', 'blue']
            verifylist = [('tags', ['red', 'blue'])]
            expected_args = ['red', 'blue', 'green']
        else:
            arglist = ['--no-tag']
            verifylist = [('no_tag', True)]
            expected_args = []
        arglist.append(self._security_group.name)
        verifylist.append(('group', self._security_group.name))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertTrue(self.network_client.update_security_group.called)
        self.network_client.set_tags.assert_called_once_with(self._security_group, tests_utils.CompareBySet(expected_args))
        self.assertIsNone(result)

    def test_set_with_tags(self):
        self._test_set_tags(with_tags=True)

    def test_set_with_no_tag(self):
        self._test_set_tags(with_tags=False)