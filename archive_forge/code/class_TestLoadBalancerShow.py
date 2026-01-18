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
class TestLoadBalancerShow(TestLoadBalancer):

    def setUp(self):
        super().setUp()
        self.api_mock.load_balancer_show.return_value = {'loadbalancer': self.lb_info}
        lb_client = self.app.client_manager
        lb_client.load_balancer = self.api_mock
        self.cmd = load_balancer.ShowLoadBalancer(self.app, None)

    def test_load_balancer_show(self):
        arglist = [self._lb.id]
        verifylist = [('loadbalancer', self._lb.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.load_balancer_show.assert_called_with(lb_id=self._lb.id)