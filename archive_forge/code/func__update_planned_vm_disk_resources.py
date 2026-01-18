import platform
from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils.compute import migrationutils
from os_win.utils.compute import vmutils
def _update_planned_vm_disk_resources(self, conn_v2_local, planned_vm, vm_name, disk_paths_remote):
    updated_resource_setting_data = []
    sasds = _wqlutils.get_element_associated_class(self._compat_conn, self._CIM_RES_ALLOC_SETTING_DATA_CLASS, element_uuid=planned_vm.Name)
    for sasd in sasds:
        if sasd.ResourceType == 17 and sasd.ResourceSubType == 'Microsoft:Hyper-V:Physical Disk Drive' and sasd.HostResource:
            old_disk_path = sasd.HostResource[0]
            new_disk_path = disk_paths_remote.pop(sasd.path().RelPath)
            LOG.debug('Replacing host resource %(old_disk_path)s with %(new_disk_path)s on planned VM %(vm_name)s', {'old_disk_path': old_disk_path, 'new_disk_path': new_disk_path, 'vm_name': vm_name})
            sasd.HostResource = [new_disk_path]
            updated_resource_setting_data.append(sasd.GetText_(1))
    LOG.debug('Updating remote planned VM disk paths for VM: %s', vm_name)
    vsmsvc = conn_v2_local.Msvm_VirtualSystemManagementService()[0]
    res_settings, job_path, ret_val = vsmsvc.ModifyResourceSettings(ResourceSettings=updated_resource_setting_data)
    self._jobutils.check_ret_val(ret_val, job_path)