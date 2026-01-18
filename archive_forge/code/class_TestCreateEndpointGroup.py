from unittest import mock
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import endpoint_group
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestCreateEndpointGroup(TestEndpointGroup, common.TestCreateVPNaaS):

    def setUp(self):
        super(TestCreateEndpointGroup, self).setUp()
        self.networkclient.create_vpn_endpoint_group = mock.Mock(return_value=_endpoint_group)
        self.mocked = self.networkclient.create_vpn_endpoint_group
        self.cmd = endpoint_group.CreateEndpointGroup(self.app, self.namespace)

    def _update_expect_response(self, request, response):
        """Set expected request and response

        :param request
            A dictionary of request body(dict of verifylist)
        :param response
            A OrderedDict of request body
        """
        self.neutronclient.create_endpoint_group.return_value = {self.res: dict(response)}
        osc_utils.find_project.return_value.id = response['tenant_id']
        self.data = _generate_data(ordered_dict=response)
        self.ordered_data = tuple((response[column] for column in self.ordered_columns))

    def _set_all_params_cidr(self, args={}):
        name = args.get('name') or 'my-name'
        description = args.get('description') or 'my-desc'
        endpoint_type = args.get('type') or 'cidr'
        endpoints = args.get('endpoints') or ['10.0.0.0/24', '20.0.0.0/24']
        tenant_id = args.get('project_id') or 'my-tenant'
        arglist = ['--description', description, '--type', endpoint_type, '--value', '10.0.0.0/24', '--value', '20.0.0.0/24', '--project', tenant_id, name]
        verifylist = [('description', description), ('type', endpoint_type), ('endpoints', endpoints), ('project', tenant_id), ('name', name)]
        return (arglist, verifylist)

    def _test_create_with_all_params_cidr(self, args={}):
        arglist, verifylist = self._set_all_params_cidr(args)
        request, response = _generate_req_and_res(verifylist)
        self._update_expect_response(request, response)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.check_results(headers, data, request)

    def test_create_with_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_with_all_params_cidr(self):
        self._test_create_with_all_params_cidr()