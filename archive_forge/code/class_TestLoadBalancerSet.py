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
class TestLoadBalancerSet(TestLoadBalancer):

    def setUp(self):
        super().setUp()
        lb_client = self.app.client_manager
        lb_client.load_balancer = self.api_mock
        self.cmd = load_balancer.SetLoadBalancer(self.app, None)

    @mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
    def test_load_balancer_set(self, mock_attrs):
        qos_policy_id = uuidutils.generate_uuid()
        mock_attrs.return_value = {'loadbalancer_id': self._lb.id, 'name': 'new_name', 'vip_qos_policy_id': qos_policy_id}
        arglist = [self._lb.id, '--name', 'new_name', '--vip-qos-policy-id', qos_policy_id]
        verifylist = [('loadbalancer', self._lb.id), ('name', 'new_name'), ('vip_qos_policy_id', qos_policy_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.load_balancer_set.assert_called_with(self._lb.id, json={'loadbalancer': {'name': 'new_name', 'vip_qos_policy_id': qos_policy_id}})

    @mock.patch('osc_lib.utils.wait_for_status')
    @mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
    def test_load_balancer_set_wait(self, mock_attrs, mock_wait):
        qos_policy_id = uuidutils.generate_uuid()
        mock_attrs.return_value = {'loadbalancer_id': self._lb.id, 'name': 'new_name', 'vip_qos_policy_id': qos_policy_id}
        arglist = [self._lb.id, '--name', 'new_name', '--vip-qos-policy-id', qos_policy_id, '--wait']
        verifylist = [('loadbalancer', self._lb.id), ('name', 'new_name'), ('vip_qos_policy_id', qos_policy_id), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.load_balancer_set.assert_called_with(self._lb.id, json={'loadbalancer': {'name': 'new_name', 'vip_qos_policy_id': qos_policy_id}})
        mock_wait.assert_called_once_with(status_f=mock.ANY, res_id=self.lb_info['id'], sleep_time=mock.ANY, status_field='provisioning_status')

    @mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
    def test_load_balancer_set_tag(self, mock_attrs):
        self.api_mock.load_balancer_show.return_value = {'tags': ['foo']}
        mock_attrs.return_value = {'loadbalancer_id': self._lb.id, 'tags': ['bar']}
        arglist = [self._lb.id, '--tag', 'bar']
        verifylist = [('loadbalancer', self._lb.id), ('tags', ['bar'])]
        try:
            parsed_args = self.check_parser(self.cmd, arglist, verifylist)
            self.cmd.take_action(parsed_args)
        except Exception as e:
            self.fail('%s raised unexpectedly' % e)
        self.api_mock.load_balancer_set.assert_called_once()
        kwargs = self.api_mock.load_balancer_set.mock_calls[0][2]
        tags = kwargs['json']['loadbalancer']['tags']
        self.assertEqual(2, len(tags))
        self.assertIn('foo', tags)
        self.assertIn('bar', tags)

    @mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
    def test_load_balancer_set_tag_no_tag(self, mock_attrs):
        self.api_mock.load_balancer_show.return_value = {'tags': ['foo']}
        mock_attrs.return_value = {'loadbalancer_id': self._lb.id, 'tags': ['bar']}
        arglist = [self._lb.id, '--tag', 'bar', '--no-tag']
        verifylist = [('loadbalancer', self._lb.id), ('tags', ['bar'])]
        try:
            parsed_args = self.check_parser(self.cmd, arglist, verifylist)
            self.cmd.take_action(parsed_args)
        except Exception as e:
            self.fail('%s raised unexpectedly' % e)
        self.api_mock.load_balancer_set.assert_called_once_with(self._lb.id, json={'loadbalancer': {'tags': ['bar']}})

    @mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
    def test_load_balancer_remove_qos_policy(self, mock_attrs):
        mock_attrs.return_value = {'loadbalancer_id': self._lb.id, 'vip_qos_policy_id': None}
        arglist = [self._lb.id, '--vip-qos-policy-id', 'None']
        verifylist = [('loadbalancer', self._lb.id), ('vip_qos_policy_id', 'None')]
        try:
            parsed_args = self.check_parser(self.cmd, arglist, verifylist)
            self.cmd.take_action(parsed_args)
        except Exception as e:
            self.fail('%s raised unexpectedly' % e)