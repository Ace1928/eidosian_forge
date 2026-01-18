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
class TestLoadBalancerDelete(TestLoadBalancer):

    def setUp(self):
        super().setUp()
        self.cmd = load_balancer.DeleteLoadBalancer(self.app, None)

    def test_load_balancer_delete(self):
        arglist = [self._lb.id]
        verifylist = [('loadbalancer', self._lb.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.load_balancer_delete.assert_called_with(lb_id=self._lb.id)

    @mock.patch('osc_lib.utils.wait_for_delete')
    def test_load_balancer_delete_wait(self, mock_wait):
        arglist = [self._lb.id, '--wait']
        verifylist = [('loadbalancer', self._lb.id), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.load_balancer_delete.assert_called_with(lb_id=self._lb.id)
        mock_wait.assert_called_once_with(manager=mock.ANY, res_id=self.lb_info['id'], sleep_time=mock.ANY, status_field='provisioning_status')

    def test_load_balancer_delete_failure(self):
        arglist = ['unknown_lb']
        verifylist = [('loadbalancer', 'unknown_lb')]
        self.api_mock.load_balancer_list.return_value = {'loadbalancers': []}
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertNotCalled(self.api_mock.load_balancer_delete)