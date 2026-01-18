from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.load_balancers import BoundLoadBalancer, PrivateNet
from ..module_utils.vendor.hcloud.networks import BoundNetwork
class AnsibleHCloudLoadBalancerNetwork(AnsibleHCloud):
    represent = 'hcloud_load_balancer_network'
    hcloud_network: BoundNetwork | None = None
    hcloud_load_balancer: BoundLoadBalancer | None = None
    hcloud_load_balancer_network: PrivateNet | None = None

    def _prepare_result(self):
        return {'network': to_native(self.hcloud_network.name), 'load_balancer': to_native(self.hcloud_load_balancer.name), 'ip': to_native(self.hcloud_load_balancer_network.ip)}

    def _get_load_balancer_and_network(self):
        try:
            self.hcloud_network = self._client_get_by_name_or_id('networks', self.module.params.get('network'))
            self.hcloud_load_balancer = self._client_get_by_name_or_id('load_balancers', self.module.params.get('load_balancer'))
            self.hcloud_load_balancer_network = None
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    def _get_load_balancer_network(self):
        for private_net in self.hcloud_load_balancer.private_net:
            if private_net.network.id == self.hcloud_network.id:
                self.hcloud_load_balancer_network = private_net

    def _create_load_balancer_network(self):
        params = {'network': self.hcloud_network}
        if self.module.params.get('ip') is not None:
            params['ip'] = self.module.params.get('ip')
        if not self.module.check_mode:
            try:
                self.hcloud_load_balancer.attach_to_network(**params).wait_until_finished()
            except HCloudException as exception:
                self.fail_json_hcloud(exception)
        self._mark_as_changed()
        self._get_load_balancer_and_network()
        self._get_load_balancer_network()

    def present_load_balancer_network(self):
        self._get_load_balancer_and_network()
        self._get_load_balancer_network()
        if self.hcloud_load_balancer_network is None:
            self._create_load_balancer_network()

    def delete_load_balancer_network(self):
        self._get_load_balancer_and_network()
        self._get_load_balancer_network()
        if self.hcloud_load_balancer_network is not None and self.hcloud_load_balancer is not None:
            if not self.module.check_mode:
                try:
                    self.hcloud_load_balancer.detach_from_network(self.hcloud_load_balancer_network.network).wait_until_finished()
                    self._mark_as_changed()
                except HCloudException as exception:
                    self.fail_json_hcloud(exception)
        self.hcloud_load_balancer_network = None

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(network={'type': 'str', 'required': True}, load_balancer={'type': 'str', 'required': True}, ip={'type': 'str'}, state={'choices': ['absent', 'present'], 'default': 'present'}, **super().base_module_arguments()), supports_check_mode=True)