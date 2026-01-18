from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.load_balancers import (
class AnsibleHCloudLoadBalancer(AnsibleHCloud):
    represent = 'hcloud_load_balancer'
    hcloud_load_balancer: BoundLoadBalancer | None = None

    def _prepare_result(self):
        private_ipv4_address = None if len(self.hcloud_load_balancer.private_net) == 0 else to_native(self.hcloud_load_balancer.private_net[0].ip)
        return {'id': to_native(self.hcloud_load_balancer.id), 'name': to_native(self.hcloud_load_balancer.name), 'ipv4_address': to_native(self.hcloud_load_balancer.public_net.ipv4.ip), 'ipv6_address': to_native(self.hcloud_load_balancer.public_net.ipv6.ip), 'private_ipv4_address': private_ipv4_address, 'load_balancer_type': to_native(self.hcloud_load_balancer.load_balancer_type.name), 'algorithm': to_native(self.hcloud_load_balancer.algorithm.type), 'location': to_native(self.hcloud_load_balancer.location.name), 'labels': self.hcloud_load_balancer.labels, 'delete_protection': self.hcloud_load_balancer.protection['delete'], 'disable_public_interface': False if self.hcloud_load_balancer.public_net.enabled else True}

    def _get_load_balancer(self):
        try:
            if self.module.params.get('id') is not None:
                self.hcloud_load_balancer = self.client.load_balancers.get_by_id(self.module.params.get('id'))
            else:
                self.hcloud_load_balancer = self.client.load_balancers.get_by_name(self.module.params.get('name'))
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    def _create_load_balancer(self):
        self.module.fail_on_missing_params(required_params=['name', 'load_balancer_type'])
        try:
            params = {'name': self.module.params.get('name'), 'algorithm': LoadBalancerAlgorithm(type=self.module.params.get('algorithm', 'round_robin')), 'load_balancer_type': self.client.load_balancer_types.get_by_name(self.module.params.get('load_balancer_type')), 'labels': self.module.params.get('labels')}
            if self.module.params.get('location') is None and self.module.params.get('network_zone') is None:
                self.module.fail_json(msg='one of the following is required: location, network_zone')
            elif self.module.params.get('location') is not None and self.module.params.get('network_zone') is None:
                params['location'] = self.client.locations.get_by_name(self.module.params.get('location'))
            elif self.module.params.get('location') is None and self.module.params.get('network_zone') is not None:
                params['network_zone'] = self.module.params.get('network_zone')
            if not self.module.check_mode:
                resp = self.client.load_balancers.create(**params)
                resp.action.wait_until_finished(max_retries=1000)
                delete_protection = self.module.params.get('delete_protection')
                if delete_protection is not None:
                    self._get_load_balancer()
                    self.hcloud_load_balancer.change_protection(delete=delete_protection).wait_until_finished()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)
        self._mark_as_changed()
        self._get_load_balancer()

    def _update_load_balancer(self):
        try:
            labels = self.module.params.get('labels')
            if labels is not None and labels != self.hcloud_load_balancer.labels:
                if not self.module.check_mode:
                    self.hcloud_load_balancer.update(labels=labels)
                self._mark_as_changed()
            delete_protection = self.module.params.get('delete_protection')
            if delete_protection is not None and delete_protection != self.hcloud_load_balancer.protection['delete']:
                if not self.module.check_mode:
                    self.hcloud_load_balancer.change_protection(delete=delete_protection).wait_until_finished()
                self._mark_as_changed()
            self._get_load_balancer()
            disable_public_interface = self.module.params.get('disable_public_interface')
            if disable_public_interface is not None and disable_public_interface != (not self.hcloud_load_balancer.public_net.enabled):
                if not self.module.check_mode:
                    if disable_public_interface is True:
                        self.hcloud_load_balancer.disable_public_interface().wait_until_finished()
                    else:
                        self.hcloud_load_balancer.enable_public_interface().wait_until_finished()
                self._mark_as_changed()
            load_balancer_type = self.module.params.get('load_balancer_type')
            if load_balancer_type is not None and self.hcloud_load_balancer.load_balancer_type.name != load_balancer_type:
                new_load_balancer_type = self.client.load_balancer_types.get_by_name(load_balancer_type)
                if not new_load_balancer_type:
                    self.module.fail_json(msg='unknown load balancer type')
                if not self.module.check_mode:
                    self.hcloud_load_balancer.change_type(load_balancer_type=new_load_balancer_type).wait_until_finished(max_retries=1000)
                self._mark_as_changed()
            algorithm = self.module.params.get('algorithm')
            if algorithm is not None and self.hcloud_load_balancer.algorithm.type != algorithm:
                self.hcloud_load_balancer.change_algorithm(algorithm=LoadBalancerAlgorithm(type=algorithm)).wait_until_finished()
                self._mark_as_changed()
            self._get_load_balancer()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    def present_load_balancer(self):
        self._get_load_balancer()
        if self.hcloud_load_balancer is None:
            self._create_load_balancer()
        else:
            self._update_load_balancer()

    def delete_load_balancer(self):
        try:
            self._get_load_balancer()
            if self.hcloud_load_balancer is not None:
                if not self.module.check_mode:
                    self.client.load_balancers.delete(self.hcloud_load_balancer)
                self._mark_as_changed()
            self.hcloud_load_balancer = None
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(id={'type': 'int'}, name={'type': 'str'}, load_balancer_type={'type': 'str'}, algorithm={'choices': ['round_robin', 'least_connections'], 'default': 'round_robin'}, location={'type': 'str'}, network_zone={'type': 'str'}, labels={'type': 'dict'}, delete_protection={'type': 'bool'}, disable_public_interface={'type': 'bool', 'default': False}, state={'choices': ['absent', 'present'], 'default': 'present'}, **super().base_module_arguments()), required_one_of=[['id', 'name']], mutually_exclusive=[['location', 'network_zone']], supports_check_mode=True)