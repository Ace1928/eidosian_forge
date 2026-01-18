from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.networks import BoundNetwork, NetworkSubnet
class AnsibleHCloudSubnetwork(AnsibleHCloud):
    represent = 'hcloud_subnetwork'
    hcloud_network: BoundNetwork | None = None
    hcloud_subnetwork: NetworkSubnet | None = None

    def _prepare_result(self):
        return {'network': to_native(self.hcloud_network.name), 'ip_range': to_native(self.hcloud_subnetwork.ip_range), 'type': to_native(self.hcloud_subnetwork.type), 'network_zone': to_native(self.hcloud_subnetwork.network_zone), 'gateway': self.hcloud_subnetwork.gateway, 'vswitch_id': self.hcloud_subnetwork.vswitch_id}

    def _get_network(self):
        try:
            self.hcloud_network = self._client_get_by_name_or_id('networks', self.module.params.get('network'))
            self.hcloud_subnetwork = None
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    def _get_subnetwork(self):
        subnet_ip_range = self.module.params.get('ip_range')
        for subnetwork in self.hcloud_network.subnets:
            if subnetwork.ip_range == subnet_ip_range:
                self.hcloud_subnetwork = subnetwork

    def _create_subnetwork(self):
        params = {'ip_range': self.module.params.get('ip_range'), 'type': self.module.params.get('type'), 'network_zone': self.module.params.get('network_zone')}
        if self.module.params.get('type') == NetworkSubnet.TYPE_VSWITCH:
            self.module.fail_on_missing_params(required_params=['vswitch_id'])
            params['vswitch_id'] = self.module.params.get('vswitch_id')
        if not self.module.check_mode:
            try:
                self.hcloud_network.add_subnet(subnet=NetworkSubnet(**params)).wait_until_finished()
            except HCloudException as exception:
                self.fail_json_hcloud(exception)
        self._mark_as_changed()
        self._get_network()
        self._get_subnetwork()

    def present_subnetwork(self):
        self._get_network()
        self._get_subnetwork()
        if self.hcloud_subnetwork is None:
            self._create_subnetwork()

    def delete_subnetwork(self):
        self._get_network()
        self._get_subnetwork()
        if self.hcloud_subnetwork is not None and self.hcloud_network is not None:
            if not self.module.check_mode:
                try:
                    self.hcloud_network.delete_subnet(self.hcloud_subnetwork).wait_until_finished()
                except HCloudException as exception:
                    self.fail_json_hcloud(exception)
            self._mark_as_changed()
        self.hcloud_subnetwork = None

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(network={'type': 'str', 'required': True}, network_zone={'type': 'str', 'required': True}, type={'type': 'str', 'required': True, 'choices': ['server', 'cloud', 'vswitch']}, ip_range={'type': 'str', 'required': True}, vswitch_id={'type': 'int'}, state={'choices': ['absent', 'present'], 'default': 'present'}, **super().base_module_arguments()), supports_check_mode=True)