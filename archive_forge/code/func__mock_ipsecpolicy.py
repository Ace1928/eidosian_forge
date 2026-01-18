from unittest import mock
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import ipsecpolicy
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
def _mock_ipsecpolicy(*args, **kwargs):
    self.networkclient.find_vpn_ipsec_policy.assert_called_once_with(self.resource['id'], ignore_missing=False)
    return {'id': args[0]}