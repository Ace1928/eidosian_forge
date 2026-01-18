from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _pre_shutdown_action(self, entity):
    vm_service = self._service.vm_service(entity.id)
    self.__suspend_shutdown_common(vm_service)
    if entity.status in [otypes.VmStatus.SUSPENDED, otypes.VmStatus.PAUSED]:
        vm_service.start()
        self._wait_for_UP(vm_service)
    return vm_service.get()