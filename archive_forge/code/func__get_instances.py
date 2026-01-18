from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def _get_instances(self, deployment):
    dep_tree = self._build_hierarchy(deployment.properties.dependencies)
    vms = self._get_dependencies(dep_tree, resource_type='Microsoft.Compute/virtualMachines')
    vms_and_nics = [(vm, self._get_dependencies(vm['children'], 'Microsoft.Network/networkInterfaces')) for vm in vms]
    vms_and_ips = [(vm['dep'], self._nic_to_public_ips_instance(nics)) for vm, nics in vms_and_nics]
    return [dict(vm_name=vm.resource_name, ips=[self._get_ip_dict(ip) for ip in ips]) for vm, ips in vms_and_ips if len(ips) > 0]