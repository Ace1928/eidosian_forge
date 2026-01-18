from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _wait_for_vm_disks(self, vm_service):
    disks_service = self._connection.system_service().disks_service()
    for da in vm_service.disk_attachments_service().list():
        disk_service = disks_service.disk_service(da.disk.id)
        wait(service=disk_service, condition=lambda disk: disk.status == otypes.DiskStatus.OK if disk.storage_type == otypes.DiskStorageType.IMAGE else True, wait=self.param('wait'), timeout=self.param('timeout'))