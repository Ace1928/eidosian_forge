import platform
from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils.compute import migrationutils
from os_win.utils.compute import vmutils
def _get_disk_data(self, vm_name, vmutils_remote, disk_path_mapping):
    disk_paths = {}
    phys_disk_resources = vmutils_remote.get_vm_disks(vm_name)[1]
    for disk in phys_disk_resources:
        rasd_rel_path = disk.path().RelPath
        serial = disk.ElementName
        disk_paths[rasd_rel_path] = disk_path_mapping[serial]
    return disk_paths