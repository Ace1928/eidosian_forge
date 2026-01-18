from unittest import mock
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import ipsecpolicy
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestIPSecPolicy(test_fakes.TestNeutronClientOSCV2):

    def check_results(self, headers, data, exp_req, is_list=False):
        if is_list:
            req_body = {self.res_plural: list(exp_req)}
        else:
            req_body = exp_req
        self.mocked.assert_called_once_with(**req_body)
        self.assertEqual(self.ordered_headers, headers)
        self.assertEqual(self.ordered_data, data)

    def setUp(self):
        super(TestIPSecPolicy, self).setUp()

        def _mock_ipsecpolicy(*args, **kwargs):
            self.networkclient.find_vpn_ipsec_policy.assert_called_once_with(self.resource['id'], ignore_missing=False)
            return {'id': args[0]}
        self.networkclient.find_vpn_ipsec_policy.side_effect = mock.Mock(side_effect=_mock_ipsecpolicy)
        osc_utils.find_project = mock.Mock()
        osc_utils.find_project.id = _ipsecpolicy['project_id']
        self.res = 'ipsecpolicy'
        self.res_plural = 'ipsecpolicies'
        self.resource = _ipsecpolicy
        self.headers = ('ID', 'Name', 'Authentication Algorithm', 'Encapsulation Mode', 'Transform Protocol', 'Encryption Algorithm', 'Perfect Forward Secrecy (PFS)', 'Description', 'Project', 'Lifetime')
        self.data = _generate_data()
        self.ordered_headers = ('Authentication Algorithm', 'Description', 'Encapsulation Mode', 'Encryption Algorithm', 'ID', 'Lifetime', 'Name', 'Perfect Forward Secrecy (PFS)', 'Project', 'Transform Protocol')
        self.ordered_data = (_ipsecpolicy['auth_algorithm'], _ipsecpolicy['description'], _ipsecpolicy['encapsulation_mode'], _ipsecpolicy['encryption_algorithm'], _ipsecpolicy['id'], _ipsecpolicy['lifetime'], _ipsecpolicy['name'], _ipsecpolicy['pfs'], _ipsecpolicy['project_id'], _ipsecpolicy['transform_protocol'])
        self.ordered_columns = ('auth_algorithm', 'description', 'encapsulation_mode', 'encryption_algorithm', 'id', 'lifetime', 'name', 'pfs', 'project_id', 'transform_protocol')