from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class ClusterNetworksModule(BaseModule):

    def __init__(self, network_id, cluster_network, *args, **kwargs):
        super(ClusterNetworksModule, self).__init__(*args, **kwargs)
        self._network_id = network_id
        self._cluster_network = cluster_network
        self._old_usages = []
        self._cluster_network_entity = get_entity(self._service.network_service(network_id))
        if self._cluster_network_entity is not None:
            self._old_usages = self._cluster_network_entity.usages

    def build_entity(self):
        return otypes.Network(id=self._network_id, name=self._module.params['name'], required=self._cluster_network.get('required'), display=self._cluster_network.get('display'), usages=list(set([otypes.NetworkUsage(usage) for usage in ['display', 'gluster', 'migration', 'default_route'] if self._cluster_network.get(usage, False)] + self._old_usages)) if self._cluster_network.get('display') is not None or self._cluster_network.get('gluster') is not None or self._cluster_network.get('migration') is not None or (self._cluster_network.get('default_route') is not None) else None)

    def update_check(self, entity):
        return equal(self._cluster_network.get('required'), entity.required) and equal(self._cluster_network.get('display'), entity.display) and all((x in [str(usage) for usage in getattr(entity, 'usages', []) if usage != otypes.NetworkUsage.VM and usage != otypes.NetworkUsage.MANAGEMENT] for x in [usage for usage in ['display', 'gluster', 'migration', 'default_route'] if self._cluster_network.get(usage, False)]))