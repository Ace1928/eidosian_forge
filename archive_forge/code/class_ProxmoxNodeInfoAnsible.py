from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (
class ProxmoxNodeInfoAnsible(ProxmoxAnsible):

    def get_nodes(self):
        nodes = self.proxmox_api.nodes.get()
        return nodes