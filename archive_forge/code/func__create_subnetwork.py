from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.networks import BoundNetwork, NetworkSubnet
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