from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _wait_for_UP(self, vm_service):
    wait(service=vm_service, condition=lambda vm: vm.status == otypes.VmStatus.UP, wait=self.param('wait'), timeout=self.param('timeout'))