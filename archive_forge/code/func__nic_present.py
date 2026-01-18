from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, TaskError, vmware_argument_spec, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def _nic_present(self):
    changed = False
    diff = {'before': {}, 'after': {}}
    force = self.params['force']
    label = self.params['label']
    mac_address = self.params['mac_address']
    network_name = self.params['network_name']
    switch = self.params['switch']
    vlan_id = self.params['vlan_id']
    vm_obj = self.get_vm()
    if not vm_obj:
        self.module.fail_json(msg='could not find vm: {0}'.format(self.params['name']))
    if self.params['device_type'] == 'pvrdma':
        if int(vm_obj.config.version.split('vmx-')[-1]) > 19 or int(vm_obj.config.version.split('vmx-')[-1]) == 13:
            self.params['pvrdma_device_protocol'] = None
        else:
            if self.params['pvrdma_device_protocol'] and self.params['pvrdma_device_protocol'] not in ['rocev1', 'rocev2']:
                self.module.fail_json(msg="Valid values of parameter 'pvrdma_device_protocol' are 'rocev1', 'rocev2' for VM with hardware version >= 14 and <= 19.")
            if self.params['pvrdma_device_protocol'] is None:
                self.params['pvrdma_device_protocol'] = 'rocev2'
    network_obj = self._get_network_object(vm_obj)
    nic_info, nic_obj_lst = self._get_nics_from_vm(vm_obj)
    label_lst = [d.get('label') for d in nic_info]
    mac_addr_lst = [d.get('mac_address') for d in nic_info]
    vlan_id_lst = [d.get('vlan_id') for d in nic_info]
    network_name_lst = [d.get('network_name') for d in nic_info]
    if vlan_id and vlan_id in vlan_id_lst or ((network_name and network_name in network_name_lst) and (not mac_address) and (not label) and (not force)):
        for nic in nic_info:
            diff['before'].update({nic.get('mac_address'): copy.copy(nic)})
            diff['after'].update({nic.get('mac_address'): copy.copy(nic)})
        return (diff, changed, nic_info)
    if not network_obj and (network_name or vlan_id):
        self.module.fail_json(msg='unable to find specified network_name/vlan_id ({0}), check parameters'.format(network_name or vlan_id))
    for nic in nic_info:
        diff['before'].update({nic.get('mac_address'): copy.copy(nic)})
    if mac_address and mac_address in mac_addr_lst or (label and label in label_lst):
        for nic_obj in nic_obj_lst:
            if mac_address and nic_obj.macAddress == mac_address or (label and label == nic_obj.deviceInfo.label):
                device_spec = self._new_nic_spec(vm_obj, nic_obj)
        if self.module.check_mode:
            for nic in nic_info:
                nic_mac = nic.get('mac_address')
                nic_label = nic.get('label')
                if nic_mac == mac_address or nic_label == label:
                    diff['after'][nic_mac] = copy.deepcopy(nic)
                    diff['after'][nic_mac].update({'switch': switch or nic['switch']})
                    if network_obj:
                        diff['after'][nic_mac].update({'vlan_id': self._get_vlanid_from_network(network_obj), 'network_name': network_obj.name})
                else:
                    diff['after'].update({nic_mac: copy.deepcopy(nic)})
    if (not mac_address or mac_address not in mac_addr_lst) and (not label or label not in label_lst):
        device_spec = self._new_nic_spec(vm_obj, None)
        device_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.add
        if self.module.check_mode:
            diff['after'] = copy.deepcopy(diff['before'])
            nic_mac = mac_address
            if not nic_mac:
                nic_mac = 'AA:BB:CC:DD:EE:FF'
            if not label:
                label = 'check_mode_adapter'
            diff['after'].update({nic_mac: {'vlan_id': self._get_vlanid_from_network(network_obj), 'network_name': network_obj.name, 'label': label, 'mac_address': nic_mac, 'unit_number': 40000}})
    if self.module.check_mode:
        network_info = [diff['after'][i] for i in diff['after']]
        if diff['after'] != diff['before']:
            changed = True
        return (diff, changed, network_info)
    if not self.module.check_mode:
        try:
            task = vm_obj.ReconfigVM_Task(vim.vm.ConfigSpec(deviceChange=[device_spec]))
            wait_for_task(task)
        except (vim.fault.InvalidDeviceSpec, vim.fault.RestrictedVersion) as e:
            self.module.fail_json(msg='failed to reconfigure guest', detail=e.msg)
        except TaskError as task_e:
            self.module.fail_json(msg=to_native(task_e))
        if task.info.state == 'error':
            self.module.fail_json(msg='failed to reconfigure guest', detail=task.info.error.msg)
        vm_obj = self.get_vm()
        network_info, nic_obj_lst = self._get_nics_from_vm(vm_obj)
        for nic in network_info:
            diff['after'].update({nic.get('mac_address'): copy.copy(nic)})
        if diff['after'] != diff['before']:
            changed = True
        return (diff, changed, network_info)