from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
class VmwareHbaScan(PyVmomi):

    def __init__(self, module):
        super(VmwareHbaScan, self).__init__(module)

    def scan(self):
        esxi_host_name = self.params.get('esxi_hostname', None)
        cluster_name = self.params.get('cluster_name', None)
        rescan_hba = self.params.get('rescan_hba', bool)
        refresh_storage = self.params.get('refresh_storage', bool)
        rescan_vmfs = self.params.get('rescan_vmfs', bool)
        hosts = self.get_all_host_objs(cluster_name=cluster_name, esxi_host_name=esxi_host_name)
        results = dict(changed=True, result=dict())
        if not hosts:
            self.module.fail_json(msg='Failed to find any hosts.')
        for host in hosts:
            results['result'][host.name] = dict()
            if rescan_hba is True:
                host.configManager.storageSystem.RescanAllHba()
            if refresh_storage is True:
                host.configManager.storageSystem.RefreshStorageSystem()
            if rescan_vmfs is True:
                host.configManager.storageSystem.RescanVmfs()
            results['result'][host.name]['rescaned_hba'] = rescan_hba
            results['result'][host.name]['refreshed_storage'] = refresh_storage
            results['result'][host.name]['rescaned_vmfs'] = rescan_vmfs
        self.module.exit_json(**results)