from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def _get_vgpu_VirtualDevice_object(self, vm_obj, vgpu_prfl):
    """
        Get current vGPU profile object of virtual machine
        Args:
            vm_obj: Managed object of virtual machine
            vgpu_prfl: vGPU profile name
        Returns: vGPU profile name of virtual machine
        """
    for vgpu_VirtualDevice_obj in vm_obj.config.hardware.device:
        if hasattr(vgpu_VirtualDevice_obj.backing, 'vgpu'):
            if vgpu_VirtualDevice_obj.backing.vgpu == vgpu_prfl:
                return vgpu_VirtualDevice_obj
    return None