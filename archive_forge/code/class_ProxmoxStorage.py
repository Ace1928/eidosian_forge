from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (
class ProxmoxStorage:

    def __init__(self, storage):
        self.storage = storage
        if 'shared' in self.storage:
            self.storage['shared'] = proxmox_to_ansible_bool(self.storage['shared'])
        if 'content' in self.storage:
            self.storage['content'] = self.storage['content'].split(',')
        if 'nodes' in self.storage:
            self.storage['nodes'] = self.storage['nodes'].split(',')
        if 'prune-backups' in storage:
            options = storage['prune-backups'].split(',')
            self.storage['prune-backups'] = dict()
            for option in options:
                k, v = option.split('=')
                self.storage['prune-backups'][k] = v