import copy
from unittest import mock
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7rule
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestL7RuleSet(TestL7Rule):

    def setUp(self):
        super().setUp()
        self.cmd = l7rule.SetL7Rule(self.app, None)

    @mock.patch('octaviaclient.osc.v2.utils.get_l7rule_attrs')
    def test_l7rule_set(self, mock_attrs):
        mock_attrs.return_value = {'admin_state_up': False, 'l7policy_id': self._l7po.id, 'l7rule_id': self._l7ru.id}
        arglist = [self._l7po.id, self._l7ru.id, '--disable']
        verifylist = [('l7policy', self._l7po.id), ('l7rule', self._l7ru.id), ('disable', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.l7rule_set.assert_called_with(l7rule_id=self._l7ru.id, l7policy_id=self._l7po.id, json={'rule': {'admin_state_up': False}})

    @mock.patch('osc_lib.utils.wait_for_status')
    @mock.patch('octaviaclient.osc.v2.utils.get_l7rule_attrs')
    def test_l7rule_set_wait(self, mock_attrs, mock_wait):
        mock_attrs.return_value = {'admin_state_up': False, 'l7policy_id': self._l7po.id, 'l7rule_id': self._l7ru.id}
        arglist = [self._l7po.id, self._l7ru.id, '--disable', '--wait']
        verifylist = [('l7policy', self._l7po.id), ('l7rule', self._l7ru.id), ('disable', True), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.l7rule_set.assert_called_with(l7rule_id=self._l7ru.id, l7policy_id=self._l7po.id, json={'rule': {'admin_state_up': False}})
        mock_wait.assert_called_once_with(status_f=mock.ANY, res_id=self._l7po.id, sleep_time=mock.ANY, status_field='provisioning_status')

    @mock.patch('octaviaclient.osc.v2.utils.get_l7rule_attrs')
    def test_l7rule_set_tag(self, mock_attrs):
        self.api_mock.l7rule_show.return_value = {'tags': ['foo']}
        mock_attrs.return_value = {'l7policy_id': self._l7po.id, 'l7rule_id': self._l7ru.id, 'tags': ['bar']}
        arglist = [self._l7po.id, self._l7ru.id, '--tag', 'bar']
        verifylist = [('l7policy', self._l7po.id), ('l7rule', self._l7ru.id), ('tags', ['bar'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.l7rule_set.assert_called_once()
        kwargs = self.api_mock.l7rule_set.mock_calls[0][2]
        tags = kwargs['json']['rule']['tags']
        self.assertEqual(2, len(tags))
        self.assertIn('foo', tags)
        self.assertIn('bar', tags)

    @mock.patch('octaviaclient.osc.v2.utils.get_l7rule_attrs')
    def test_l7rule_set_tag_no_tag(self, mock_attrs):
        self.api_mock.l7rule_show.return_value = {'tags': ['foo']}
        mock_attrs.return_value = {'l7policy_id': self._l7po.id, 'l7rule_id': self._l7ru.id, 'tags': ['bar']}
        arglist = [self._l7po.id, self._l7ru.id, '--tag', 'bar', '--no-tag']
        verifylist = [('l7policy', self._l7po.id), ('l7rule', self._l7ru.id), ('tags', ['bar']), ('no_tag', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.l7rule_set.assert_called_once_with(l7policy_id=self._l7po.id, l7rule_id=self._l7ru.id, json={'rule': {'tags': ['bar']}})