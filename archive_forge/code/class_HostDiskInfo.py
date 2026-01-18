from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
class HostDiskInfo(PyVmomi):
    """Class to return host disk info"""

    def __init__(self, module):
        super(HostDiskInfo, self).__init__(module)
        cluster_name = self.params.get('cluster_name', None)
        esxi_host_name = self.params.get('esxi_hostname', None)
        self.hosts = self.get_all_host_objs(cluster_name=cluster_name, esxi_host_name=esxi_host_name)
        if not self.hosts:
            self.module.fail_json(msg='Failed to find host system.')

    def gather_host_disk_info(self):
        hosts_disk_info = {}
        for host in self.hosts:
            host_disk_info = []
            storage_system = host.configManager.storageSystem.storageDeviceInfo
            lun_lookup = {}
            for lun in storage_system.multipathInfo.lun:
                key = lun.lun
                paths = []
                for path in lun.path:
                    paths.append(path.name)
                lun_lookup[key] = paths
            for disk in storage_system.scsiLun:
                canonical_name = disk.canonicalName
                try:
                    capacity = int(disk.capacity.block * disk.capacity.blockSize / 1048576)
                except AttributeError:
                    capacity = 0
                try:
                    device_path = disk.devicePath
                except AttributeError:
                    device_path = ''
                device_type = disk.deviceType
                display_name = disk.displayName
                disk_uid = disk.key
                device_ctd_list = lun_lookup[disk_uid]
                disk_dict = {'capacity_mb': capacity, 'device_path': device_path, 'device_type': device_type, 'display_name': display_name, 'disk_uid': disk_uid, 'device_ctd_list': device_ctd_list, 'canonical_name': canonical_name}
                host_disk_info.append(disk_dict)
            hosts_disk_info[host.name] = host_disk_info
        return hosts_disk_info