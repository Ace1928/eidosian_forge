from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def _add_vgpu_profile_to_vm(self, vm_obj, vgpu_profile_name, vgpu_prfl):
    """
        Add vGPU profile of virtual machine
        Args:
            vm_obj: Managed object of virtual machine
            vgpu_profile_name: vGPU profile object name from ESXi server list
            vgpu_prfl: vGPU profile name
        Returns: Operation results
        """
    changed = False
    failed = False
    vm_current_vgpu_profile = self._get_vgpu_profile_in_the_vm(vm_obj)
    if self.params['force'] or vgpu_prfl not in vm_current_vgpu_profile:
        vgpu_p = vgpu_profile_name.vgpu
        backing = vim.VirtualPCIPassthroughVmiopBackingInfo(vgpu=vgpu_p)
        summary = 'NVIDIA GRID vGPU ' + vgpu_prfl
        deviceInfo = vim.Description(summary=summary, label='PCI device 0')
        hba_object = vim.VirtualPCIPassthrough(backing=backing, deviceInfo=deviceInfo)
        new_device_config = vim.VirtualDeviceConfigSpec(device=hba_object)
        new_device_config.operation = 'add'
        vmConfigSpec = vim.vm.ConfigSpec()
        vmConfigSpec.deviceChange = [new_device_config]
        vmConfigSpec.memoryReservationLockedToMax = True
        try:
            task = vm_obj.ReconfigVM_Task(spec=vmConfigSpec)
            wait_for_task(task)
            changed = True
        except Exception as exc:
            failed = True
            self.module.fail_json(msg="Failed to add vGPU Profile '%s' to vm %s." % (vgpu_prfl, vm_obj.name), detail=exc.msg)
    else:
        return (changed, failed)
    return (changed, failed)