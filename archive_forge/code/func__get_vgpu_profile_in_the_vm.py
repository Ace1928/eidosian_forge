from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def _get_vgpu_profile_in_the_vm(self, vm_obj):
    """
        Get vGPU profile object of virtual machine
        Args:
            vm_obj: Managed object of virtual machine
        Returns: vGPU profile name
        """
    vm_current_vgpu_profile = []
    for vgpu_VirtualDevice_obj in vm_obj.config.hardware.device:
        if hasattr(vgpu_VirtualDevice_obj.backing, 'vgpu'):
            vm_current_vgpu_profile.append(vgpu_VirtualDevice_obj.backing.vgpu)
    return vm_current_vgpu_profile