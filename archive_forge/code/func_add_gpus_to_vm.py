import logging
from pyVim.task import WaitForTask
from pyVmomi import vim
def add_gpus_to_vm(pyvmomi_sdk_provider, vm_name: str, gpu_cards: list, is_dynamic_pci_passthrough):
    """
    This function helps to add a list of gpu to a VM by PCI passthrough. Steps:
    1. Power off the VM if it is not at the off state.
    2. Construct a reconfigure spec and reconfigure the VM.
    3. Power on the VM.
    """
    vm_obj = pyvmomi_sdk_provider.get_pyvmomi_obj([vim.VirtualMachine], vm_name)
    if vm_obj.runtime.powerState == vim.VirtualMachinePowerState.poweredOn:
        logger.debug(f'Power off VM {vm_name}...')
        WaitForTask(vm_obj.PowerOffVM_Task())
        logger.debug(f'VM {vm_name} is power off. Done.')
    config_spec = vim.vm.ConfigSpec()
    config_spec.extraConfig = [vim.option.OptionValue(key='pciPassthru.64bitMMIOSizeGB', value='64'), vim.option.OptionValue(key='pciPassthru.use64bitMMIO', value='TRUE')]
    config_spec.memoryReservationLockedToMax = True
    config_spec.cpuHotAddEnabled = False
    config_spec.deviceChange = []
    pci_passthroughs = vm_obj.environmentBrowser.QueryConfigTarget(host=None).pciPassthrough
    id_to_pci_passthru_info = {item.pciDevice.id: item for item in pci_passthroughs}
    key = -100
    for gpu_card in gpu_cards:
        pci_id = gpu_card.pciId
        custom_label = gpu_card.customLabel
        pci_passthru_info = id_to_pci_passthru_info[pci_id]
        device_id = pci_passthru_info.pciDevice.deviceId
        vendor_id = pci_passthru_info.pciDevice.vendorId
        backing = None
        if is_dynamic_pci_passthrough:
            logger.info(f'Plugin GPU card - Id {pci_id} deviceId {device_id} vendorId {vendor_id} customLabel {custom_label} into VM {vm_name}')
            allowed_device = vim.VirtualPCIPassthroughAllowedDevice(vendorId=vendor_id, deviceId=device_id)
            backing = vim.VirtualPCIPassthroughDynamicBackingInfo(allowedDevice=[allowed_device], customLabel=custom_label, assignedId=str(device_id))
        else:
            logger.info(f'Plugin GPU card {pci_id} into VM {vm_name}')
            backing = vim.VirtualPCIPassthroughDeviceBackingInfo(deviceId=hex(pci_passthru_info.pciDevice.deviceId % 2 ** 16).lstrip('0x'), id=pci_id, systemId=pci_passthru_info.systemId, vendorId=pci_passthru_info.pciDevice.vendorId, deviceName=pci_passthru_info.pciDevice.deviceName)
        gpu = vim.VirtualPCIPassthrough(key=key, backing=backing)
        device_change = vim.vm.device.VirtualDeviceSpec(operation='add', device=gpu)
        config_spec.deviceChange.append(device_change)
        key += 1
    WaitForTask(vm_obj.ReconfigVM_Task(spec=config_spec))
    logger.debug(f'Power on VM {vm_name}...')
    WaitForTask(vm_obj.PowerOnVM_Task())
    logger.debug(f'VM {vm_name} is power on. Done.')