import argparse
import copy
import itertools
from unittest import mock
from osc_lib import exceptions
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import load_balancer
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('osc_lib.utils.wait_for_status')
def _test_load_balancer_unset_param_wait(self, param, mock_wait):
    self.api_mock.load_balancer_set.reset_mock()
    ref_body = {'loadbalancer': {param: None}}
    arg_param = param.replace('_', '-') if '_' in param else param
    arglist = [self._lb.id, '--%s' % arg_param, '--wait']
    verifylist = [('loadbalancer', self._lb.id), ('wait', True)]
    for ref_param in self.PARAMETERS:
        verifylist.append((ref_param, param == ref_param))
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.load_balancer_set.assert_called_once_with(self._lb.id, json=ref_body)
    mock_wait.assert_called_once_with(status_f=mock.ANY, res_id=self.lb_info['id'], sleep_time=mock.ANY, status_field='provisioning_status')