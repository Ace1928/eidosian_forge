from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec, ProxmoxAnsible)
def _container_mp_get(self, vm, vmid):
    cfg = self.vmconfig(vm, vmid).get()
    mountpoints = {}
    for key, value in cfg.items():
        if key.startswith('mp'):
            mountpoints[key] = value
    return mountpoints