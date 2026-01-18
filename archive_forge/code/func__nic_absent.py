from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, TaskError, vmware_argument_spec, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def _nic_absent(self, network_params=None):
    changed = False
    diff = {'before': {}, 'after': {}}
    if network_params:
        mac_address = network_params['mac_address']
    else:
        mac_address = self.params['mac_address']
    device_spec = None
    vm_obj = self.get_vm()
    if not vm_obj:
        self.module.fail_json(msg='could not find vm: {0}'.format(self.params['name']))
    nic_info, nic_obj_lst = self._get_nics_from_vm(vm_obj)
    for nic in nic_info:
        diff['before'].update({nic['mac_address']: copy.copy(nic)})
    network_info = copy.deepcopy(nic_info)
    for nic_obj in nic_obj_lst:
        if nic_obj.macAddress == mac_address:
            if self.module.check_mode:
                changed = True
                for nic in nic_info:
                    if nic.get('mac_address') != nic_obj.macAddress:
                        diff['after'].update({nic['mac_address']: copy.copy(nic)})
                network_info = [nic for nic in nic_info if nic.get('mac_address') != nic_obj.macAddress]
                return (diff, changed, network_info)
            device_spec = vim.vm.device.VirtualDeviceSpec(device=nic_obj, operation=vim.vm.device.VirtualDeviceSpec.Operation.remove)
            break
    if not device_spec:
        diff['after'] = diff['before']
        return (diff, changed, network_info)
    try:
        task = vm_obj.ReconfigVM_Task(vim.vm.ConfigSpec(deviceChange=[device_spec]))
        wait_for_task(task)
    except (vim.fault.InvalidDeviceSpec, vim.fault.RestrictedVersion) as e:
        self.module.fail_json(msg='failed to reconfigure guest', detail=e.msg)
    if task.info.state == 'error':
        self.module.fail_json(msg='failed to reconfigure guest', detail=task.info.error.msg)
    vm_obj = self.get_vm()
    nic_info, nic_obj_lst = self._get_nics_from_vm(vm_obj)
    for nic in nic_info:
        diff['after'].update({nic.get('mac_address'): copy.copy(nic)})
    network_info = nic_info
    if diff['after'] != diff['before']:
        changed = True
    return (diff, changed, network_info)