from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
from collections import defaultdict
def _get_external_gateway_network_name(self):
    network_name_or_id = self.params['network']
    if self.params['external_gateway_info']:
        network_name_or_id = self.params['external_gateway_info']['network']
    return network_name_or_id