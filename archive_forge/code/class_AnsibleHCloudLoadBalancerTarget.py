from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.load_balancers import (
from ..module_utils.vendor.hcloud.servers import BoundServer
class AnsibleHCloudLoadBalancerTarget(AnsibleHCloud):
    represent = 'hcloud_load_balancer_target'
    hcloud_load_balancer: BoundLoadBalancer | None = None
    hcloud_load_balancer_target: LoadBalancerTarget | None = None
    hcloud_server: BoundServer | None = None

    def _prepare_result(self):
        result = {'type': to_native(self.hcloud_load_balancer_target.type), 'load_balancer': to_native(self.hcloud_load_balancer.name), 'use_private_ip': self.hcloud_load_balancer_target.use_private_ip}
        if self.hcloud_load_balancer_target.type == 'server':
            result['server'] = to_native(self.hcloud_load_balancer_target.server.name)
        elif self.hcloud_load_balancer_target.type == 'label_selector':
            result['label_selector'] = to_native(self.hcloud_load_balancer_target.label_selector.selector)
        elif self.hcloud_load_balancer_target.type == 'ip':
            result['ip'] = to_native(self.hcloud_load_balancer_target.ip.ip)
        return result

    def _get_load_balancer_and_target(self):
        try:
            self.hcloud_load_balancer = self._client_get_by_name_or_id('load_balancers', self.module.params.get('load_balancer'))
            if self.module.params.get('type') == 'server':
                self.hcloud_server = self._client_get_by_name_or_id('servers', self.module.params.get('server'))
            self.hcloud_load_balancer_target = None
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    def _get_load_balancer_target(self):
        for target in self.hcloud_load_balancer.targets:
            if self.module.params.get('type') == 'server' and target.type == 'server':
                if target.server.id == self.hcloud_server.id:
                    self.hcloud_load_balancer_target = target
            elif self.module.params.get('type') == 'label_selector' and target.type == 'label_selector':
                if target.label_selector.selector == self.module.params.get('label_selector'):
                    self.hcloud_load_balancer_target = target
            elif self.module.params.get('type') == 'ip' and target.type == 'ip':
                if target.ip.ip == self.module.params.get('ip'):
                    self.hcloud_load_balancer_target = target

    def _create_load_balancer_target(self):
        params = {'target': None}
        if self.module.params.get('type') == 'server':
            self.module.fail_on_missing_params(required_params=['server'])
            params['target'] = LoadBalancerTarget(type=self.module.params.get('type'), server=self.hcloud_server, use_private_ip=self.module.params.get('use_private_ip'))
        elif self.module.params.get('type') == 'label_selector':
            self.module.fail_on_missing_params(required_params=['label_selector'])
            params['target'] = LoadBalancerTarget(type=self.module.params.get('type'), label_selector=LoadBalancerTargetLabelSelector(selector=self.module.params.get('label_selector')), use_private_ip=self.module.params.get('use_private_ip'))
        elif self.module.params.get('type') == 'ip':
            self.module.fail_on_missing_params(required_params=['ip'])
            params['target'] = LoadBalancerTarget(type=self.module.params.get('type'), ip=LoadBalancerTargetIP(ip=self.module.params.get('ip')), use_private_ip=False)
        if not self.module.check_mode:
            try:
                self.hcloud_load_balancer.add_target(**params).wait_until_finished()
            except APIException as exception:
                if exception.code == 'locked' or exception.code == 'conflict':
                    self._create_load_balancer_target()
                else:
                    self.fail_json_hcloud(exception)
            except HCloudException as exception:
                self.fail_json_hcloud(exception)
        self._mark_as_changed()
        self._get_load_balancer_and_target()
        self._get_load_balancer_target()

    def present_load_balancer_target(self):
        self._get_load_balancer_and_target()
        self._get_load_balancer_target()
        if self.hcloud_load_balancer_target is None:
            self._create_load_balancer_target()

    def delete_load_balancer_target(self):
        self._get_load_balancer_and_target()
        self._get_load_balancer_target()
        if self.hcloud_load_balancer_target is not None and self.hcloud_load_balancer is not None:
            if not self.module.check_mode:
                target = None
                if self.module.params.get('type') == 'server':
                    self.module.fail_on_missing_params(required_params=['server'])
                    target = LoadBalancerTarget(type=self.module.params.get('type'), server=self.hcloud_server)
                elif self.module.params.get('type') == 'label_selector':
                    self.module.fail_on_missing_params(required_params=['label_selector'])
                    target = LoadBalancerTarget(type=self.module.params.get('type'), label_selector=LoadBalancerTargetLabelSelector(selector=self.module.params.get('label_selector')), use_private_ip=self.module.params.get('use_private_ip'))
                elif self.module.params.get('type') == 'ip':
                    self.module.fail_on_missing_params(required_params=['ip'])
                    target = LoadBalancerTarget(type=self.module.params.get('type'), ip=LoadBalancerTargetIP(ip=self.module.params.get('ip')), use_private_ip=False)
                try:
                    self.hcloud_load_balancer.remove_target(target).wait_until_finished()
                except HCloudException as exception:
                    self.fail_json_hcloud(exception)
            self._mark_as_changed()
        self.hcloud_load_balancer_target = None

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(type={'type': 'str', 'required': True, 'choices': ['server', 'label_selector', 'ip']}, load_balancer={'type': 'str', 'required': True}, server={'type': 'str'}, label_selector={'type': 'str'}, ip={'type': 'str'}, use_private_ip={'type': 'bool', 'default': False}, state={'choices': ['absent', 'present'], 'default': 'present'}, **super().base_module_arguments()), supports_check_mode=True)