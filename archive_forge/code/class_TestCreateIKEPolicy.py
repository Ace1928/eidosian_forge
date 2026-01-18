from unittest import mock
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import ikepolicy
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestCreateIKEPolicy(TestIKEPolicy, common.TestCreateVPNaaS):

    def setUp(self):
        super(TestCreateIKEPolicy, self).setUp()
        self.networkclient.create_vpn_ike_policy = mock.Mock(return_value=_ikepolicy)
        self.mocked = self.networkclient.create_vpn_ike_policy
        self.cmd = ikepolicy.CreateIKEPolicy(self.app, self.namespace)

    def _update_expect_response(self, request, response):
        """Set expected request and response

        :param request
            A dictionary of request body(dict of verifylist)
        :param response
            A OrderedDict of request body
        """
        self.networkclient.create_vpn_ikepolicy.return_value = response
        osc_utils.find_project.return_value.id = response['project_id']
        self.data = _generate_data(ordered_dict=response)
        self.ordered_data = tuple((response[column] for column in self.ordered_columns))

    def _set_all_params(self, args={}):
        name = args.get('name') or 'my-name'
        description = args.get('description') or 'my-desc'
        auth_algorithm = args.get('auth_algorithm') or 'sha1'
        encryption_algorithm = args.get('encryption_algorithm') or 'aes-128'
        phase1_negotiation_mode = args.get('phase1_negotiation_mode') or 'main'
        ike_version = args.get('ike_version') or 'v1'
        pfs = args.get('pfs') or 'group5'
        tenant_id = args.get('tenant_id') or 'my-tenant'
        arglist = ['--description', description, '--auth-algorithm', auth_algorithm, '--encryption-algorithm', encryption_algorithm, '--phase1-negotiation-mode', phase1_negotiation_mode, '--ike-version', ike_version, '--pfs', pfs, '--project', tenant_id, name]
        verifylist = [('description', description), ('auth_algorithm', auth_algorithm), ('encryption_algorithm', encryption_algorithm), ('phase1_negotiation_mode', phase1_negotiation_mode), ('ike_version', ike_version), ('pfs', pfs), ('project', tenant_id), ('name', name)]
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
        self._test_create_with_all_params({'name': 'new_ikepolicy'})

    def test_create_with_all_params_aggressive_mode(self):
        self._test_create_with_all_params({'phase1_negotiation_mode': 'aggressive'})