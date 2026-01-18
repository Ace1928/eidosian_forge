from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (
class ProxmoxStorageInfoAnsible(ProxmoxAnsible):

    def get_storage(self, storage):
        try:
            storage = self.proxmox_api.storage.get(storage)
        except Exception:
            self.module.fail_json(msg="Storage '%s' does not exist" % storage)
        return ProxmoxStorage(storage)

    def get_storages(self, type=None):
        storages = self.proxmox_api.storage.get(type=type)
        storages = [ProxmoxStorage(storage) for storage in storages]
        return storages