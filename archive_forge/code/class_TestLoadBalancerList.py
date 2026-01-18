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
class TestLoadBalancerList(TestLoadBalancer):

    def setUp(self):
        super().setUp()
        self.datalist = (tuple((attr_consts.LOADBALANCER_ATTRS[k] for k in self.columns)),)
        self.cmd = load_balancer.ListLoadBalancer(self.app, None)

    def test_load_balancer_list_no_options(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.load_balancer_list.assert_called_with()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_load_balancer_list_with_name(self):
        arglist = ['--name', 'rainbarrel']
        verifylist = [('name', 'rainbarrel')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.load_balancer_list.assert_called_with(name='rainbarrel')
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    @mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
    def test_load_balancer_list_with_network(self, mock_client):
        mock_client.return_value = {'vip_network_id': self._lb.vip_network_id}
        arglist = ['--vip-network-id', self._lb.vip_network_id]
        verify_list = [('vip_network_id', self._lb.vip_network_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verify_list)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    @mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
    def test_load_balancer_list_with_subnet(self, mock_client):
        mock_client.return_value = {'vip_subnet_id': self._lb.vip_subnet_id}
        arglist = ['--vip-subnet-id', self._lb.vip_subnet_id]
        verify_list = [('vip_subnet_id', self._lb.vip_subnet_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verify_list)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    @mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
    def test_load_balancer_list_with_qos_policy(self, mock_client):
        mock_client.return_value = {'vip_qos_policy_id': self._lb.vip_qos_policy_id}
        arglist = ['--vip-qos-policy-id', self._lb.vip_qos_policy_id]
        verify_list = [('vip_qos_policy_id', self._lb.vip_qos_policy_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verify_list)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    @mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
    def test_load_balancer_list_with_provisioning_status(self, mock_client):
        mock_client.return_value = {'provisioning_status': self._lb.provisioning_status}
        arglist = ['--provisioning-status', 'active']
        verify_list = [('provisioning_status', 'ACTIVE')]
        parsed_args = self.check_parser(self.cmd, arglist, verify_list)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    @mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
    def test_load_balancer_list_with_operating_status(self, mock_client):
        mock_client.return_value = {'operating_status': self._lb.operating_status}
        arglist = ['--operating-status', 'ONLiNE']
        verify_list = [('operating_status', 'ONLINE')]
        parsed_args = self.check_parser(self.cmd, arglist, verify_list)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    @mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
    def test_load_balancer_list_with_provider(self, mock_client):
        mock_client.return_value = {'provider': self._lb.provider}
        arglist = ['--provider', 'octavia']
        verify_list = [('provider', 'octavia')]
        parsed_args = self.check_parser(self.cmd, arglist, verify_list)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    @mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
    def test_load_balancer_list_with_flavor(self, mock_client):
        mock_client.return_value = {'flavor_id': self._lb.flavor_id}
        arglist = ['--flavor', self._lb.flavor_id]
        verify_list = [('flavor', self._lb.flavor_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verify_list)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    @mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
    def test_load_balancer_list_with_tags(self, mock_client):
        mock_client.return_value = {'tags': self._lb.tags}
        arglist = ['--tags', ','.join(self._lb.tags)]
        verify_list = [('tags', self._lb.tags)]
        parsed_args = self.check_parser(self.cmd, arglist, verify_list)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    @mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
    def test_load_balancer_list_with_any_tags(self, mock_client):
        mock_client.return_value = {'tags': self._lb.tags}
        arglist = ['--any-tags', ','.join(self._lb.tags)]
        verify_list = [('any_tags', self._lb.tags)]
        parsed_args = self.check_parser(self.cmd, arglist, verify_list)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    @mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
    def test_load_balancer_list_with_not_tags(self, mock_client):
        mock_client.return_value = {'tags': self._lb.tags[0]}
        arglist = ['--any-tags', ','.join(self._lb.tags)]
        verify_list = [('any_tags', self._lb.tags)]
        parsed_args = self.check_parser(self.cmd, arglist, verify_list)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    @mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
    def test_load_balancer_list_with_not_any_tags(self, mock_client):
        mock_client.return_value = {'tags': self._lb.tags[0]}
        arglist = ['--not-any-tags', ','.join(self._lb.tags)]
        verify_list = [('not_any_tags', self._lb.tags)]
        parsed_args = self.check_parser(self.cmd, arglist, verify_list)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))