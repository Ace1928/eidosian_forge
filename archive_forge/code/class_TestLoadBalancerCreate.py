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
class TestLoadBalancerCreate(TestLoadBalancer):

    def setUp(self):
        super().setUp()
        self.api_mock.load_balancer_create.return_value = {'loadbalancer': self.lb_info}
        lb_client = self.app.client_manager
        lb_client.load_balancer = self.api_mock
        self.cmd = load_balancer.CreateLoadBalancer(self.app, None)

    @mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
    def test_load_balancer_create(self, mock_client):
        mock_client.return_value = self.lb_info
        arglist = ['--name', self._lb.name, '--vip-network-id', self._lb.vip_network_id, '--project', self._lb.project_id, '--flavor', self._lb.flavor_id]
        verifylist = [('name', self._lb.name), ('vip_network_id', self._lb.vip_network_id), ('project', self._lb.project_id), ('flavor', self._lb.flavor_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.load_balancer_create.assert_called_with(json={'loadbalancer': self.lb_info})

    @mock.patch('osc_lib.utils.wait_for_status')
    @mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
    def test_load_balancer_create_wait(self, mock_client, mock_wait):
        mock_client.return_value = self.lb_info
        self.api_mock.load_balancer_show.return_value = self.lb_info
        arglist = ['--name', self._lb.name, '--vip-network-id', self._lb.vip_network_id, '--project', self._lb.project_id, '--flavor', self._lb.flavor_id, '--wait']
        verifylist = [('name', self._lb.name), ('vip_network_id', self._lb.vip_network_id), ('project', self._lb.project_id), ('flavor', self._lb.flavor_id), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.load_balancer_create.assert_called_with(json={'loadbalancer': self.lb_info})
        mock_wait.assert_called_once_with(status_f=mock.ANY, res_id=self.lb_info['id'], sleep_time=mock.ANY, status_field='provisioning_status')

    @mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
    def test_load_balancer_create_with_qos_policy(self, mock_client):
        qos_policy_id = 'qos_id'
        lb_info = copy.deepcopy(self.lb_info)
        lb_info.update({'vip_qos_policy_id': qos_policy_id})
        mock_client.return_value = lb_info
        arglist = ['--name', self._lb.name, '--vip-network-id', self._lb.vip_network_id, '--project', self._lb.project_id, '--vip-qos-policy-id', qos_policy_id]
        verifylist = [('name', self._lb.name), ('vip_network_id', self._lb.vip_network_id), ('project', self._lb.project_id), ('vip_qos_policy_id', qos_policy_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.load_balancer_create.assert_called_with(json={'loadbalancer': lb_info})

    @mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
    def test_load_balancer_create_with_additional_vips(self, mock_client):
        mock_client.return_value = self.lb_info
        arglist = ['--name', self._lb.name, '--vip-subnet-id', self._lb.vip_subnet_id, '--project', self._lb.project_id, '--additional-vip', 'subnet-id={},ip-address={}'.format(self._lb.additional_vips[0]['subnet_id'], self._lb.additional_vips[0]['ip_address']), '--additional-vip', 'subnet-id={},ip-address={}'.format(self._lb.additional_vips[1]['subnet_id'], self._lb.additional_vips[1]['ip_address'])]
        verifylist = [('name', self._lb.name), ('vip_subnet_id', self._lb.vip_subnet_id), ('project', self._lb.project_id), ('additional_vip', ['subnet-id={},ip-address={}'.format(self._lb.additional_vips[0]['subnet_id'], self._lb.additional_vips[0]['ip_address']), 'subnet-id={},ip-address={}'.format(self._lb.additional_vips[1]['subnet_id'], self._lb.additional_vips[1]['ip_address'])])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.load_balancer_create.assert_called_with(json={'loadbalancer': self.lb_info})

    @mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
    def test_load_balancer_create_with_provider(self, mock_client):
        provider = 'foobar'
        lb_info = copy.deepcopy(self.lb_info)
        lb_info.update({'provider': provider})
        mock_client.return_value = lb_info
        arglist = ['--name', self._lb.name, '--vip-network-id', self._lb.vip_network_id, '--project', self._lb.project_id, '--provider', provider]
        verifylist = [('name', self._lb.name), ('vip_network_id', self._lb.vip_network_id), ('project', self._lb.project_id), ('provider', provider)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.load_balancer_create.assert_called_with(json={'loadbalancer': lb_info})

    @mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
    def test_load_balancer_create_with_tags(self, mock_client):
        lb_info = copy.deepcopy(self.lb_info)
        lb_info.update({'tags': self._lb.tags})
        mock_client.return_value = lb_info
        arglist = ['--name', self._lb.name, '--vip-network-id', self._lb.vip_network_id, '--project', self._lb.project_id, '--tag', self._lb.tags[0], '--tag', self._lb.tags[1]]
        verifylist = [('name', self._lb.name), ('vip_network_id', self._lb.vip_network_id), ('project', self._lb.project_id), ('tags', self._lb.tags)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.load_balancer_create.assert_called_with(json={'loadbalancer': lb_info})

    @mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
    def test_load_balancer_create_missing_args(self, mock_client):
        attrs_list = self.lb_info
        args = ('vip_subnet_id', 'vip_network_id', 'vip_port_id')
        for a in args:
            attrs_list[a] = ''
        for n in range(len(args) + 1):
            for comb in itertools.combinations(args, n):
                filtered_attrs = {k: v for k, v in attrs_list.items() if k not in comb}
                filtered_attrs['wait'] = False
                mock_client.return_value = filtered_attrs
                parsed_args = argparse.Namespace(**filtered_attrs)
                if not any((k in filtered_attrs for k in args)) or all((k in filtered_attrs for k in ('vip_network_id', 'vip_port_id'))):
                    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
                else:
                    try:
                        self.cmd.take_action(parsed_args)
                    except exceptions.CommandError as e:
                        self.fail('%s raised unexpectedly' % e)