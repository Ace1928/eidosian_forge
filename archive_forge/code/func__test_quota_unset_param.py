import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import quota
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('octaviaclient.osc.v2.utils.get_resource_id')
def _test_quota_unset_param(self, param, mock_get_resource):
    self.api_mock.quota_set.reset_mock()
    mock_get_resource.return_value = self._qt.project_id
    arg_param = param.replace('_', '-') if '_' in param else param
    arglist = [self._qt.project_id, '--%s' % arg_param]
    ref_body = {'quota': {param: None}}
    verifylist = [('project', self._qt.project_id)]
    for ref_param in self.PARAMETERS:
        verifylist.append((ref_param, param == ref_param))
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.quota_set.assert_called_once_with(self._qt.project_id, json=ref_body)