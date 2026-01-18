from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
class VMwareVmVssDvsMigrate(object):

    def __init__(self, module):
        self.module = module
        self.content = connect_to_api(module)
        self.vm = None
        self.vm_name = module.params['vm_name']
        self.dvportgroup_name = module.params['dvportgroup_name']

    def process_state(self):
        vm_nic_states = {'absent': self.migrate_network_adapter_vds, 'present': self.state_exit_unchanged}
        vm_nic_states[self.check_vm_network_state()]()

    def find_dvspg_by_name(self):
        vmware_distributed_port_group = get_all_objs(self.content, [vim.dvs.DistributedVirtualPortgroup])
        for dvspg in vmware_distributed_port_group:
            if dvspg.name == self.dvportgroup_name:
                return dvspg
        return None

    def find_vm_by_name(self):
        virtual_machines = get_all_objs(self.content, [vim.VirtualMachine])
        for vm in virtual_machines:
            if vm.name == self.vm_name:
                return vm
        return None

    def migrate_network_adapter_vds(self):
        vm_configspec = vim.vm.ConfigSpec()
        nic = vim.vm.device.VirtualEthernetCard.DistributedVirtualPortBackingInfo()
        port = vim.dvs.PortConnection()
        devicespec = vim.vm.device.VirtualDeviceSpec()
        pg = self.find_dvspg_by_name()
        if pg is None:
            self.module.fail_json(msg='The standard portgroup was not found')
        dvswitch = pg.config.distributedVirtualSwitch
        port.switchUuid = dvswitch.uuid
        port.portgroupKey = pg.key
        nic.port = port
        for device in self.vm.config.hardware.device:
            if isinstance(device, vim.vm.device.VirtualEthernetCard):
                devicespec.device = device
                devicespec.operation = vim.vm.device.VirtualDeviceSpec.Operation.edit
                devicespec.device.backing = nic
                vm_configspec.deviceChange.append(devicespec)
        task = self.vm.ReconfigVM_Task(vm_configspec)
        changed, result = wait_for_task(task)
        self.module.exit_json(changed=changed, result=result)

    def state_exit_unchanged(self):
        self.module.exit_json(changed=False)

    def check_vm_network_state(self):
        try:
            self.vm = self.find_vm_by_name()
            if self.vm is None:
                self.module.fail_json(msg='A virtual machine with name %s does not exist' % self.vm_name)
            for device in self.vm.config.hardware.device:
                if isinstance(device, vim.vm.device.VirtualEthernetCard):
                    if isinstance(device.backing, vim.vm.device.VirtualEthernetCard.DistributedVirtualPortBackingInfo):
                        return 'present'
            return 'absent'
        except vmodl.RuntimeFault as runtime_fault:
            self.module.fail_json(msg=runtime_fault.msg)
        except vmodl.MethodFault as method_fault:
            self.module.fail_json(msg=method_fault.msg)