import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import flavor
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def _test_flavor_unset_param(self, param):
    self.api_mock.flavor_set.reset_mock()
    arg_param = param.replace('_', '-') if '_' in param else param
    arglist = [self._flavor.id, '--%s' % arg_param]
    ref_body = {'flavor': {param: None}}
    verifylist = [('flavor', self._flavor.id)]
    for ref_param in self.PARAMETERS:
        verifylist.append((ref_param, param == ref_param))
    print(verifylist)
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.flavor_set.assert_called_once_with(self._flavor.id, json=ref_body)