from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def __suspend_shutdown_common(self, vm_service):
    if vm_service.get().status in [otypes.VmStatus.MIGRATING, otypes.VmStatus.POWERING_UP, otypes.VmStatus.REBOOT_IN_PROGRESS, otypes.VmStatus.WAIT_FOR_LAUNCH, otypes.VmStatus.UP, otypes.VmStatus.RESTORING_STATE]:
        self._wait_for_UP(vm_service)