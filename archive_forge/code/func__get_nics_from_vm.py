from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, TaskError, vmware_argument_spec, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def _get_nics_from_vm(self, vm_obj):
    """
        return a list of dictionaries containing vm nic info and
        a list of objects
        :param vm_obj: object containing virtual machine
        :return: list of dicts and list ith nic object(s)
        :rtype: list, list
        """
    nic_info_lst = []
    nics = [nic for nic in vm_obj.config.hardware.device if isinstance(nic, vim.vm.device.VirtualEthernetCard)]
    for nic in nics:
        d_item = dict(mac_address=nic.macAddress, label=nic.deviceInfo.label, unit_number=nic.unitNumber, wake_onlan=nic.wakeOnLanEnabled, allow_guest_ctl=nic.connectable.allowGuestControl, connected=nic.connectable.connected, start_connected=nic.connectable.startConnected)
        if isinstance(nic, vim.vm.device.VirtualSriovEthernetCard):
            d_item['allow_guest_os_mtu_change'] = nic.allowGuestOSMtuChange
            if isinstance(nic.sriovBacking, vim.vm.device.VirtualSriovEthernetCard.SriovBackingInfo):
                if isinstance(nic.sriovBacking.physicalFunctionBacking, vim.vm.device.VirtualPCIPassthrough):
                    d_item['physical_function_backing'] = nic.sriovBacking.physicalFunctionBacking.id
                if isinstance(nic.sriovBacking.virtualFunctionBacking, vim.vm.device.VirtualPCIPassthrough):
                    d_item['virtual_function_backing'] = nic.sriovBacking.virtualFunctionBacking.id
        if isinstance(nic.backing, vim.vm.device.VirtualEthernetCard.DistributedVirtualPortBackingInfo):
            key = nic.backing.port.portgroupKey
            for portgroup in vm_obj.network:
                if hasattr(portgroup, 'key') and portgroup.key == key:
                    d_item['network_name'] = portgroup.name
                    d_item['switch'] = portgroup.config.distributedVirtualSwitch.name
                    break
        elif isinstance(nic.backing, vim.vm.device.VirtualEthernetCard.OpaqueNetworkBackingInfo):
            d_item['network_name'] = nic.backing.opaqueNetworkId
            d_item['switch'] = nic.backing.opaqueNetworkType
        elif isinstance(nic.backing, vim.vm.device.VirtualEthernetCard.NetworkBackingInfo):
            d_item['network_name'] = nic.backing.network.name
            d_item['vlan_id'] = self._get_vlanid_from_network(nic.backing.network)
            if isinstance(nic.backing.network, vim.Network):
                for pg in vm_obj.runtime.host.config.network.portgroup:
                    if pg.spec.name == nic.backing.network.name:
                        d_item['switch'] = pg.spec.vswitchName
                        break
        for k in self.device_helper.nic_device_type:
            if isinstance(nic, self.device_helper.nic_device_type[k]):
                d_item['device_type'] = k
                if k == 'vmxnet3':
                    continue
                else:
                    break
        if d_item['device_type'] == 'pvrdma':
            d_item['device_protocol'] = nic.deviceProtocol
        nic_info_lst.append(d_item)
    nic_info_lst = sorted(nic_info_lst, key=lambda d: d['mac_address'] if d['mac_address'] is not None else '00:00:00:00:00:00')
    return (nic_info_lst, nics)