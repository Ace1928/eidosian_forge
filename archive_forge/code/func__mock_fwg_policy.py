import copy
import re
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import firewallgroup
from neutronclient.osc.v2 import utils as v2_utils
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.fwaas import common
from neutronclient.tests.unit.osc.v2.fwaas import fakes
def _mock_fwg_policy(*args, **kwargs):
    if self.networkclient.find_firewall_group.call_count == 1:
        self.networkclient.find_firewall_group.assert_called_with(target)
    if self.networkclient.find_firewall_policy.call_count == 1:
        self.networkclient.find_firewall_policy.assert_called_with(ingress_policy)
    if self.networkclient.find_firewall_policy.call_count == 2:
        self.networkclient.find_firewall_policy.assert_called_with(egress_policy)
    return {'id': args[0]}