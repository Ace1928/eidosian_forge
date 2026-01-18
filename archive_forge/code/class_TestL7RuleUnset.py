import copy
from unittest import mock
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7rule
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestL7RuleUnset(TestL7Rule):
    PARAMETERS = ('invert', 'key')

    def setUp(self):
        super().setUp()
        self.cmd = l7rule.UnsetL7Rule(self.app, None)

    def test_l7rule_unset_invert(self):
        self._test_l7rule_unset_param('invert')

    def test_l7rule_unset_invert_wait(self):
        self._test_l7rule_unset_param_wait('invert')

    def test_l7rule_unset_key(self):
        self._test_l7rule_unset_param('key')

    def _test_l7rule_unset_param(self, param):
        self.api_mock.l7rule_set.reset_mock()
        arg_param = param.replace('_', '-') if '_' in param else param
        arglist = [self._l7po.id, self._l7ru.id, '--%s' % arg_param]
        ref_body = {'rule': {param: None}}
        verifylist = [('l7rule_id', self._l7ru.id)]
        for ref_param in self.PARAMETERS:
            verifylist.append((ref_param, param == ref_param))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.l7rule_set.assert_called_once_with(l7policy_id=self._l7po.id, l7rule_id=self._l7ru.id, json=ref_body)

    @mock.patch('osc_lib.utils.wait_for_status')
    def _test_l7rule_unset_param_wait(self, param, mock_wait):
        self.api_mock.l7rule_set.reset_mock()
        arg_param = param.replace('_', '-') if '_' in param else param
        arglist = [self._l7po.id, self._l7ru.id, '--%s' % arg_param, '--wait']
        ref_body = {'rule': {param: None}}
        verifylist = [('l7policy', self._l7po.id), ('l7rule_id', self._l7ru.id), ('wait', True)]
        for ref_param in self.PARAMETERS:
            verifylist.append((ref_param, param == ref_param))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.l7rule_set.assert_called_once_with(l7policy_id=self._l7po.id, l7rule_id=self._l7ru.id, json=ref_body)
        mock_wait.assert_called_once_with(status_f=mock.ANY, res_id=self._l7po.id, sleep_time=mock.ANY, status_field='provisioning_status')

    def test_l7rule_unset_all(self):
        self.api_mock.l7rule_set.reset_mock()
        ref_body = {'rule': {x: None for x in self.PARAMETERS}}
        arglist = [self._l7po.id, self._l7ru.id]
        for ref_param in self.PARAMETERS:
            arg_param = ref_param.replace('_', '-') if '_' in ref_param else ref_param
            arglist.append('--%s' % arg_param)
        verifylist = list(zip(self.PARAMETERS, [True] * len(self.PARAMETERS)))
        verifylist = [('l7rule_id', self._l7ru.id)] + verifylist
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.l7rule_set.assert_called_once_with(l7policy_id=self._l7po.id, l7rule_id=self._l7ru.id, json=ref_body)

    def test_l7rule_unset_none(self):
        self.api_mock.l7rule_set.reset_mock()
        arglist = [self._l7po.id, self._l7ru.id]
        verifylist = list(zip(self.PARAMETERS, [False] * len(self.PARAMETERS)))
        verifylist = [('l7rule_id', self._l7ru.id)] + verifylist
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.l7rule_set.assert_not_called()

    @mock.patch('octaviaclient.osc.v2.utils.get_l7rule_attrs')
    def test_l7rule_unset_tag(self, mock_attrs):
        self.api_mock.l7rule_show.return_value = {'tags': ['foo', 'bar']}
        mock_attrs.return_value = {'l7policy_id': self._l7po.id, 'l7rule_id': self._l7ru.id, 'tags': ['foo', 'bar']}
        arglist = [self._l7po.id, self._l7ru.id, '--tag', 'foo']
        verifylist = [('l7policy', self._l7po.id), ('l7rule_id', self._l7ru.id), ('tags', ['foo'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.l7rule_set.assert_called_once_with(l7policy_id=self._l7po.id, l7rule_id=self._l7ru.id, json={'rule': {'tags': ['bar']}})

    @mock.patch('octaviaclient.osc.v2.utils.get_l7rule_attrs')
    def test_l7rule_unset_all_tags(self, mock_attrs):
        self.api_mock.l7rule_show.return_value = {'tags': ['foo', 'bar']}
        mock_attrs.return_value = {'l7policy_id': self._l7po.id, 'l7rule_id': self._l7ru.id, 'tags': ['foo', 'bar']}
        arglist = [self._l7po.id, self._l7ru.id, '--all-tag']
        verifylist = [('l7policy', self._l7po.id), ('l7rule_id', self._l7ru.id), ('all_tag', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.l7rule_set.assert_called_once_with(l7policy_id=self._l7po.id, l7rule_id=self._l7ru.id, json={'rule': {'tags': []}})