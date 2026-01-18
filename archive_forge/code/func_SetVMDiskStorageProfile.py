from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
def SetVMDiskStorageProfile(self, vm, unit_number, controller_number, profile):
    """
        Set VM's disk storage policy profile.

        :param vm: A virtual machine object
        :type vm: VirtualMachine
        :param unit_number: virtual machine's disk unit number.
        :type unit_number: int
        :param controller_number: virtual machine's controller number.
        :type controller_number: int
        :param profile: A VMware Storage Policy profile
        :type profile: pbm.profile.Profile
        :returns: VMware task object.
        :rtype: Task
        """
    spec = vim.vm.ConfigSpec()
    profileSpec = vim.vm.DefinedProfileSpec()
    profileSpec.profileId = profile.profileId.uniqueId
    deviceSpec = vim.vm.device.VirtualDeviceSpec()
    deviceSpec.operation = vim.vm.device.VirtualDeviceSpec.Operation.edit
    disk_obj = self.GetVirtualDiskObj(vm, unit_number, controller_number)
    deviceSpec.device = disk_obj
    deviceSpec.profile = [profileSpec]
    spec.deviceChange = [deviceSpec]
    return vm.ReconfigVM_Task(spec)