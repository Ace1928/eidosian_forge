import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7policy
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
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