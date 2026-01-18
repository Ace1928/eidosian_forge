import argparse
import copy
import itertools
from unittest import mock
from osc_lib import exceptions
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import load_balancer
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestLoadBalancerStatus(TestLoadBalancer):

    def setUp(self):
        super().setUp()
        expected_res = {'statuses': {'operating_status': 'ONLINE', 'provisioning_status': 'ACTIVE'}}
        self.api_mock.load_balancer_status_show.return_value = {'statuses': expected_res['statuses']}
        lb_client = self.app.client_manager
        lb_client.load_balancer = self.api_mock
        self.cmd = load_balancer.ShowLoadBalancerStatus(self.app, None)

    def test_load_balancer_status_show(self):
        arglist = [self._lb.id]
        verifylist = [('loadbalancer', self._lb.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.load_balancer_status_show.assert_called_with(lb_id=self._lb.id)