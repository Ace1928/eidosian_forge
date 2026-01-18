from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.networks import BoundNetwork
from ..module_utils.vendor.hcloud.servers import BoundServer, PrivateNet
def _update_server_network(self):
    params = {'network': self.hcloud_network}
    alias_ips = self.module.params.get('alias_ips')
    if alias_ips is not None and sorted(self.hcloud_server_network.alias_ips) != sorted(alias_ips):
        params['alias_ips'] = alias_ips
        if not self.module.check_mode:
            try:
                self.hcloud_server.change_alias_ips(**params).wait_until_finished()
            except APIException as exception:
                self.fail_json_hcloud(exception)
        self._mark_as_changed()
    self._get_server_and_network()
    self._get_server_network()