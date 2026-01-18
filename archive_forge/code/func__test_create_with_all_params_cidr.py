from unittest import mock
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import endpoint_group
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
def _test_create_with_all_params_cidr(self, args={}):
    arglist, verifylist = self._set_all_params_cidr(args)
    request, response = _generate_req_and_res(verifylist)
    self._update_expect_response(request, response)
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    headers, data = self.cmd.take_action(parsed_args)
    self.check_results(headers, data, request)