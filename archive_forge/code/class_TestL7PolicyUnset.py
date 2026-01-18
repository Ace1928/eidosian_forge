import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7policy
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestL7PolicyUnset(TestL7Policy):
    PARAMETERS = ('name', 'description', 'redirect_http_code')

    def setUp(self):
        super().setUp()
        self.cmd = l7policy.UnsetL7Policy(self.app, None)

    def test_l7policy_unset_description(self):
        self._test_l7policy_unset_param('description')

    def test_l7policy_unset_name(self):
        self._test_l7policy_unset_param('name')

    def test_l7policy_unset_name_wait(self):
        self._test_l7policy_unset_param_wait('name')

    def test_l7policy_unset_redirect_http_code(self):
        self._test_l7policy_unset_param('redirect_http_code')

    def _test_l7policy_unset_param(self, param):
        self.api_mock.l7policy_set.reset_mock()
        arg_param = param.replace('_', '-') if '_' in param else param
        arglist = [self._l7po.id, '--%s' % arg_param]
        ref_body = {'l7policy': {param: None}}
        verifylist = [('l7policy', self._l7po.id)]
        for ref_param in self.PARAMETERS:
            verifylist.append((ref_param, param == ref_param))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.l7policy_set.assert_called_once_with(self._l7po.id, json=ref_body)

    @mock.patch('osc_lib.utils.wait_for_status')
    def _test_l7policy_unset_param_wait(self, param, mock_wait):
        self.api_mock.l7policy_set.reset_mock()
        arg_param = param.replace('_', '-') if '_' in param else param
        arglist = [self._l7po.id, '--%s' % arg_param, '--wait']
        ref_body = {'l7policy': {param: None}}
        verifylist = [('l7policy', self._l7po.id), ('wait', True)]
        for ref_param in self.PARAMETERS:
            verifylist.append((ref_param, param == ref_param))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.l7policy_set.assert_called_once_with(self._l7po.id, json=ref_body)
        mock_wait.assert_called_once_with(status_f=mock.ANY, res_id=self._l7po.id, sleep_time=mock.ANY, status_field='provisioning_status')

    def test_l7policy_unset_all(self):
        self.api_mock.l7policy_set.reset_mock()
        ref_body = {'l7policy': {x: None for x in self.PARAMETERS}}
        arglist = [self._l7po.id]
        for ref_param in self.PARAMETERS:
            arg_param = ref_param.replace('_', '-') if '_' in ref_param else ref_param
            arglist.append('--%s' % arg_param)
        verifylist = list(zip(self.PARAMETERS, [True] * len(self.PARAMETERS)))
        verifylist = [('l7policy', self._l7po.id)] + verifylist
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.l7policy_set.assert_called_once_with(self._l7po.id, json=ref_body)

    def test_l7policy_unset_none(self):
        self.api_mock.l7policy_set.reset_mock()
        arglist = [self._l7po.id]
        verifylist = list(zip(self.PARAMETERS, [False] * len(self.PARAMETERS)))
        verifylist = [('l7policy', self._l7po.id)] + verifylist
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.l7policy_set.assert_not_called()

    def test_l7policy_unset_tag(self):
        self.api_mock.l7policy_set.reset_mock()
        self.api_mock.l7policy_show.return_value = {'tags': ['foo', 'bar']}
        arglist = [self._l7po.id, '--tag', 'foo']
        verifylist = [('l7policy', self._l7po.id), ('tags', ['foo'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.l7policy_set.assert_called_once_with(self._l7po.id, json={'l7policy': {'tags': ['bar']}})

    def test_l7policy_unset_all_tag(self):
        self.api_mock.l7policy_set.reset_mock()
        self.api_mock.l7policy_show.return_value = {'tags': ['foo', 'bar']}
        arglist = [self._l7po.id, '--all-tag']
        verifylist = [('l7policy', self._l7po.id), ('all_tag', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.l7policy_set.assert_called_once_with(self._l7po.id, json={'l7policy': {'tags': []}})