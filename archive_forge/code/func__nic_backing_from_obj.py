from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, TaskError, vmware_argument_spec, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def _nic_backing_from_obj(self, network_obj):
    rv = None
    if isinstance(network_obj, vim.dvs.DistributedVirtualPortgroup):
        rv = vim.VirtualEthernetCardDistributedVirtualPortBackingInfo(port=vim.DistributedVirtualSwitchPortConnection(portgroupKey=network_obj.key, switchUuid=network_obj.config.distributedVirtualSwitch.uuid))
    elif isinstance(network_obj, vim.OpaqueNetwork):
        rv = vim.vm.device.VirtualEthernetCard.OpaqueNetworkBackingInfo(opaqueNetworkType='nsx.LogicalSwitch', opaqueNetworkId=network_obj.summary.opaqueNetworkId)
    elif isinstance(network_obj, vim.Network):
        rv = vim.vm.device.VirtualEthernetCard.NetworkBackingInfo(deviceName=network_obj.name, network=network_obj)
    return rv