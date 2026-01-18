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
class TestShowFirewallGroup(TestFirewallGroup, common.TestShowFWaaS):

    def setUp(self):
        super(TestShowFirewallGroup, self).setUp()
        self.networkclient.get_firewall_group = mock.Mock(return_value=_fwg)
        self.mocked = self.networkclient.get_firewall_group
        self.cmd = firewallgroup.ShowFirewallGroup(self.app, self.namespace)