from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, TaskError, vmware_argument_spec, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def _new_nic_spec(self, vm_obj, nic_obj=None, network_params=None):
    network = self._get_network_object(vm_obj)
    if network_params:
        connected = network_params['connected']
        device_type = network_params['device_type'].lower()
        directpath_io = network_params['directpath_io']
        guest_control = network_params['guest_control']
        label = network_params['label']
        mac_address = network_params['mac_address']
        start_connected = network_params['start_connected']
        wake_onlan = network_params['wake_onlan']
        pf_backing = network_params['physical_function_backing']
        vf_backing = network_params['virtual_function_backing']
        allow_guest_os_mtu_change = network_params['allow_guest_os_mtu_change']
    else:
        connected = self.params['connected']
        device_type = self.params['device_type'].lower()
        directpath_io = self.params['directpath_io']
        guest_control = self.params['guest_control']
        label = self.params['label']
        mac_address = self.params['mac_address']
        start_connected = self.params['start_connected']
        wake_onlan = self.params['wake_onlan']
        pf_backing = self.params['physical_function_backing']
        vf_backing = self.params['virtual_function_backing']
        allow_guest_os_mtu_change = self.params['allow_guest_os_mtu_change']
        pvrdma_device_protocol = self.params['pvrdma_device_protocol']
    if not nic_obj:
        device_obj = self.device_helper.nic_device_type[device_type]
        nic_spec = vim.vm.device.VirtualDeviceSpec(device=device_obj())
        if mac_address:
            nic_spec.device.addressType = 'manual'
            nic_spec.device.macAddress = mac_address
        if label:
            nic_spec.device.deviceInfo = vim.Description(label=label)
        if device_type == 'pvrdma' and pvrdma_device_protocol:
            nic_spec.device.deviceProtocol = pvrdma_device_protocol
    else:
        nic_spec = vim.vm.device.VirtualDeviceSpec(operation=vim.vm.device.VirtualDeviceSpec.Operation.edit, device=nic_obj)
        if label and label != nic_obj.deviceInfo.label:
            nic_spec.device.deviceInfo = vim.Description(label=label)
        if mac_address and mac_address != nic_obj.macAddress:
            nic_spec.device.addressType = 'manual'
            nic_spec.device.macAddress = mac_address
    nic_spec.device.backing = self._nic_backing_from_obj(network)
    nic_spec.device.connectable = vim.vm.device.VirtualDevice.ConnectInfo(startConnected=start_connected, allowGuestControl=guest_control, connected=connected)
    nic_spec.device.wakeOnLanEnabled = wake_onlan
    if (pf_backing is not None or vf_backing is not None) and (not isinstance(nic_spec.device, vim.vm.device.VirtualSriovEthernetCard)):
        self.module_fail_json(msg='physical_function_backing, virtual_function_backing can only be used with the sriov device type')
    if isinstance(nic_spec.device, vim.vm.device.VirtualSriovEthernetCard):
        nic_spec.device.allowGuestOSMtuChange = allow_guest_os_mtu_change
        nic_spec.device.sriovBacking = vim.vm.device.VirtualSriovEthernetCard.SriovBackingInfo()
        if pf_backing is not None:
            nic_spec.device.sriovBacking.physicalFunctionBacking = vim.vm.device.VirtualPCIPassthrough.DeviceBackingInfo()
            nic_spec.device.sriovBacking.physicalFunctionBacking.id = pf_backing
        if vf_backing is not None:
            nic_spec.device.sriovBacking.virtualFunctionBacking = vim.vm.device.VirtualPCIPassthrough.DeviceBackingInfo()
            nic_spec.device.sriovBacking.virtualFunctionBacking.id = vf_backing
    if directpath_io and (not isinstance(nic_spec.device, vim.vm.device.VirtualVmxnet3)):
        self.module.fail_json(msg='directpath_io can only be used with the vmxnet3 device type')
    if directpath_io and isinstance(nic_spec.device, vim.vm.device.VirtualVmxnet3):
        nic_spec.device.uptCompatibilityEnabled = True
    return nic_spec