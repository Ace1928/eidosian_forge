from unittest import mock
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import ipsecpolicy
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestCreateIPSecPolicy(TestIPSecPolicy, common.TestCreateVPNaaS):

    def setUp(self):
        super(TestCreateIPSecPolicy, self).setUp()
        self.networkclient.create_vpn_ipsec_policy = mock.Mock(return_value=_ipsecpolicy)
        self.mocked = self.networkclient.create_vpn_ipsec_policy
        self.cmd = ipsecpolicy.CreateIPsecPolicy(self.app, self.namespace)

    def _update_expect_response(self, request, response):
        """Set expected request and response

        :param request
            A dictionary of request body(dict of verifylist)
        :param response
            A OrderedDict of request body
        """
        self.networkclient.create_vpn_ipsec_policy.return_value = response
        osc_utils.find_project.return_value.id = response['project_id']
        self.data = _generate_data(ordered_dict=response)
        self.ordered_data = tuple((response[column] for column in self.ordered_columns))

    def _set_all_params(self, args={}):
        name = args.get('name') or 'my-name'
        auth_algorithm = args.get('auth_algorithm') or 'sha1'
        encapsulation_mode = args.get('encapsulation_mode') or 'tunnel'
        transform_protocol = args.get('transform_protocol') or 'esp'
        encryption_algorithm = args.get('encryption_algorithm') or 'aes-128'
        pfs = args.get('pfs') or 'group5'
        description = args.get('description') or 'my-desc'
        project_id = args.get('project_id') or 'my-project'
        arglist = [name, '--auth-algorithm', auth_algorithm, '--encapsulation-mode', encapsulation_mode, '--transform-protocol', transform_protocol, '--encryption-algorithm', encryption_algorithm, '--pfs', pfs, '--description', description, '--project', project_id]
        verifylist = [('name', name), ('auth_algorithm', auth_algorithm), ('encapsulation_mode', encapsulation_mode), ('transform_protocol', transform_protocol), ('encryption_algorithm', encryption_algorithm), ('pfs', pfs), ('description', description), ('project', project_id)]
        return (arglist, verifylist)

    def _test_create_with_all_params(self, args={}):
        arglist, verifylist = self._set_all_params(args)
        request, response = _generate_req_and_res(verifylist)
        self._update_expect_response(request, response)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.check_results(headers, data, request)

    def test_create_with_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_with_all_params(self):
        self._test_create_with_all_params()

    def test_create_with_all_params_name(self):
        self._test_create_with_all_params({'name': 'new_ipsecpolicy'})