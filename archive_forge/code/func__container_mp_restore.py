from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec, ProxmoxAnsible)
def _container_mp_restore(self, vm, vmid, timeout, unbind, mountpoints, vmstatus):
    self.vmconfig(vm, vmid).put(**mountpoints)
    if vmstatus == 'running':
        self.start_instance(vm, vmid, timeout)