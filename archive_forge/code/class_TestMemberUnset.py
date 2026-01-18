import copy
from unittest import mock
import osc_lib.tests.utils as osc_test_utils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import member
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestMemberUnset(TestMember):
    PARAMETERS = ('backup', 'monitor_address', 'monitor_port', 'name', 'weight')

    def setUp(self):
        super().setUp()
        self.cmd = member.UnsetMember(self.app, None)

    def test_member_unset_backup(self):
        self._test_member_unset_param('backup')

    def test_member_unset_monitor_address(self):
        self._test_member_unset_param('monitor_address')

    def test_member_unset_monitor_port(self):
        self._test_member_unset_param('monitor_port')

    def test_member_unset_name(self):
        self._test_member_unset_param('name')

    def test_member_unset_name_wait(self):
        self._test_member_unset_param_wait('name')

    def test_member_unset_weight(self):
        self._test_member_unset_param('weight')

    def _test_member_unset_param(self, param):
        self.api_mock.member_set.reset_mock()
        arg_param = param.replace('_', '-') if '_' in param else param
        arglist = [self._mem.pool_id, self._mem.id, '--%s' % arg_param]
        ref_body = {'member': {param: None}}
        verifylist = [('pool', self._mem.pool_id), ('member', self._mem.id)]
        for ref_param in self.PARAMETERS:
            verifylist.append((ref_param, param == ref_param))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.member_set.assert_called_once_with(pool_id=self._mem.pool_id, member_id=self._mem.id, json=ref_body)

    @mock.patch('osc_lib.utils.wait_for_status')
    def _test_member_unset_param_wait(self, param, mock_wait):
        self.api_mock.member_set.reset_mock()
        arg_param = param.replace('_', '-') if '_' in param else param
        arglist = [self._mem.pool_id, self._mem.id, '--%s' % arg_param, '--wait']
        ref_body = {'member': {param: None}}
        verifylist = [('pool', self._mem.pool_id), ('member', self._mem.id), ('wait', True)]
        for ref_param in self.PARAMETERS:
            verifylist.append((ref_param, param == ref_param))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.member_set.assert_called_once_with(pool_id=self._mem.pool_id, member_id=self._mem.id, json=ref_body)
        mock_wait.assert_called_once_with(status_f=mock.ANY, res_id=self._mem.id, sleep_time=mock.ANY, status_field='provisioning_status')

    def test_member_unset_all(self):
        self.api_mock.pool_set.reset_mock()
        ref_body = {'member': {x: None for x in self.PARAMETERS}}
        arglist = [self._mem.pool_id, self._mem.id]
        for ref_param in self.PARAMETERS:
            arg_param = ref_param.replace('_', '-') if '_' in ref_param else ref_param
            arglist.append('--%s' % arg_param)
        verifylist = list(zip(self.PARAMETERS, [True] * len(self.PARAMETERS)))
        verifylist = [('member', self._mem.id)] + verifylist
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.member_set.assert_called_once_with(member_id=self._mem.id, pool_id=self._mem.pool_id, json=ref_body)

    def test_member_unset_none(self):
        self.api_mock.pool_set.reset_mock()
        arglist = [self._mem.pool_id, self._mem.id]
        verifylist = list(zip(self.PARAMETERS, [False] * len(self.PARAMETERS)))
        verifylist = [('member', self._mem.id)] + verifylist
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.member_set.assert_not_called()

    @mock.patch('octaviaclient.osc.v2.utils.get_member_attrs')
    def test_member_unset_tag(self, mock_attrs):
        self.api_mock.member_show.return_value = {'tags': ['foo', 'bar']}
        mock_attrs.return_value = {'pool_id': self._mem.pool_id, 'member_id': self._mem.id, 'tags': ['bar']}
        arglist = [self._mem.pool_id, self._mem.id, '--tag', 'bar']
        verifylist = [('pool', self._mem.pool_id), ('member', self._mem.id), ('tags', ['bar'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.member_set.assert_called_once_with(pool_id=self._mem.pool_id, member_id=self._mem.id, json={'member': {'tags': ['foo']}})

    @mock.patch('octaviaclient.osc.v2.utils.get_member_attrs')
    def test_member_unset_all_tags(self, mock_attrs):
        self.api_mock.member_show.return_value = {'tags': ['foo', 'bar']}
        mock_attrs.return_value = {'pool_id': self._mem.pool_id, 'member_id': self._mem.id, 'tags': ['foo', 'bar']}
        arglist = [self._mem.pool_id, self._mem.id, '--all-tag']
        verifylist = [('pool', self._mem.pool_id), ('member', self._mem.id), ('all_tag', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.member_set.assert_called_once_with(pool_id=self._mem.pool_id, member_id=self._mem.id, json={'member': {'tags': []}})