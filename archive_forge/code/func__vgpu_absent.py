from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def _vgpu_absent(self, vm_obj):
    """
        Remove vGPU profile of virtual machine.
        Args:
            vm_obj: Managed object of virtual machine
        Returns: Operation results and vGPU facts
        """
    result = {}
    vgpu_prfl = self.params['vgpu']
    vgpu_VirtualDevice_obj = self._get_vgpu_VirtualDevice_object(vm_obj, vgpu_prfl)
    if vgpu_VirtualDevice_obj is None:
        changed = False
        failed = False
    else:
        vgpu_fact = self._gather_vgpu_profile_facts(vm_obj)
        changed, failed = self._remove_vgpu_profile_from_vm(vm_obj, vgpu_VirtualDevice_obj, vgpu_prfl)
    result = {'changed': changed, 'failed': failed, 'vgpu': vgpu_fact}
    return result