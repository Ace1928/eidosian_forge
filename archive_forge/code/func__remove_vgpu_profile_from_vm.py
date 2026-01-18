from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def _remove_vgpu_profile_from_vm(self, vm_obj, vgpu_VirtualDevice_obj, vgpu_prfl):
    """
        Remove vGPU profile of virtual machine
        Args:
            vm_obj: Managed object of virtual machine
            vgpu_VirtualDevice_obj: vGPU profile object holding its facts
            vgpu_prfl: vGPU profile name
        Returns: Operation results
        """
    changed = False
    failed = False
    vm_current_vgpu_profile = self._get_vgpu_profile_in_the_vm(vm_obj)
    if vgpu_prfl in vm_current_vgpu_profile:
        vdspec = vim.vm.device.VirtualDeviceSpec()
        vmConfigSpec = vim.vm.ConfigSpec()
        vdspec.operation = 'remove'
        vdspec.device = vgpu_VirtualDevice_obj
        vmConfigSpec.deviceChange.append(vdspec)
        try:
            task = vm_obj.ReconfigVM_Task(spec=vmConfigSpec)
            wait_for_task(task)
            changed = True
            return (changed, failed)
        except Exception as exc:
            failed = True
            self.module.fail_json(msg="Failed to delete vGPU profile '%s' from vm %s." % (vgpu_prfl, vm_obj.name), detail=exc.msg)
    return (changed, failed)