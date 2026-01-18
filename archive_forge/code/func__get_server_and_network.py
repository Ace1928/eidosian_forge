from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.networks import BoundNetwork
from ..module_utils.vendor.hcloud.servers import BoundServer, PrivateNet
def _get_server_and_network(self):
    try:
        self.hcloud_network = self._client_get_by_name_or_id('networks', self.module.params.get('network'))
        self.hcloud_server = self._client_get_by_name_or_id('servers', self.module.params.get('server'))
        self.hcloud_server_network = None
    except HCloudException as exception:
        self.fail_json_hcloud(exception)