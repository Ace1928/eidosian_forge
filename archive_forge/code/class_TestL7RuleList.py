import copy
from unittest import mock
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7rule
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestL7RuleList(TestL7Rule):

    def setUp(self):
        super().setUp()
        self.datalist = (tuple((attr_consts.L7RULE_ATTRS[k] for k in self.columns)),)
        self.cmd = l7rule.ListL7Rule(self.app, None)

    @mock.patch('octaviaclient.osc.v2.utils.get_l7rule_attrs')
    def test_l7rule_list_no_options(self, mock_attrs):
        mock_attrs.return_value = {'l7policy_id': self._l7po.id}
        arglist = [self._l7po.id]
        verifylist = [('l7policy', self._l7po.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.l7rule_list.assert_called_with(l7policy_id=self._l7po.id)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_l7rule_list_with_tags(self):
        arglist = [self._l7po.id, '--tags', 'foo,bar']
        verifylist = [('l7policy', self._l7po.id), ('tags', ['foo', 'bar'])]
        expected_attrs = {'l7policy_id': self._l7po.id, 'tags': ['foo', 'bar']}
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.l7rule_list.assert_called_with(**expected_attrs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_l7rule_list_with_any_tags(self):
        arglist = [self._l7po.id, '--any-tags', 'foo,bar']
        verifylist = [('l7policy', self._l7po.id), ('any_tags', ['foo', 'bar'])]
        expected_attrs = {'l7policy_id': self._l7po.id, 'tags-any': ['foo', 'bar']}
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.l7rule_list.assert_called_with(**expected_attrs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_l7rule_list_with_not_tags(self):
        arglist = [self._l7po.id, '--not-tags', 'foo,bar']
        verifylist = [('l7policy', self._l7po.id), ('not_tags', ['foo', 'bar'])]
        expected_attrs = {'l7policy_id': self._l7po.id, 'not-tags': ['foo', 'bar']}
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.l7rule_list.assert_called_with(**expected_attrs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_l7rule_list_with_not_any_tags(self):
        arglist = [self._l7po.id, '--not-any-tags', 'foo,bar']
        verifylist = [('l7policy', self._l7po.id), ('not_any_tags', ['foo', 'bar'])]
        expected_attrs = {'l7policy_id': self._l7po.id, 'not-tags-any': ['foo', 'bar']}
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.l7rule_list.assert_called_with(**expected_attrs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))