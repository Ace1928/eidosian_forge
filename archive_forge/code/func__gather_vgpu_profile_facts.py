from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def _gather_vgpu_profile_facts(self, vm_obj):
    """
        Gather facts about VM's vGPU profile settings
        Args:
            vm_obj: Managed object of virtual machine
        Returns: vGPU profile and its facts
        """
    vgpu_info = dict()
    for vgpu_VirtualDevice_obj in vm_obj.config.hardware.device:
        if hasattr(vgpu_VirtualDevice_obj.backing, 'vgpu'):
            vgpu_info = dict(Vgpu=vgpu_VirtualDevice_obj.backing.vgpu, Key=vgpu_VirtualDevice_obj.key, Summary=vgpu_VirtualDevice_obj.deviceInfo.summary, Label=vgpu_VirtualDevice_obj.deviceInfo.label, Unit_Number=vgpu_VirtualDevice_obj.unitNumber, Controller_Key=vgpu_VirtualDevice_obj.controllerKey)
            break
    return vgpu_info