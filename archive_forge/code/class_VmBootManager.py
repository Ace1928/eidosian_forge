from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, find_vm_by_id, wait_for_task, TaskError
class VmBootManager(PyVmomi):

    def __init__(self, module):
        super(VmBootManager, self).__init__(module)
        self.name = self.params['name']
        self.uuid = self.params['uuid']
        self.moid = self.params['moid']
        self.use_instance_uuid = self.params['use_instance_uuid']
        self.vm = None

    def _get_vm(self):
        vms = []
        if self.uuid:
            if self.use_instance_uuid:
                vm_obj = find_vm_by_id(self.content, vm_id=self.uuid, vm_id_type='instance_uuid')
            else:
                vm_obj = find_vm_by_id(self.content, vm_id=self.uuid, vm_id_type='uuid')
            if vm_obj is None:
                self.module.fail_json(msg='Failed to find the virtual machine with UUID : %s' % self.uuid)
            vms = [vm_obj]
        elif self.name:
            objects = self.get_managed_objects_properties(vim_type=vim.VirtualMachine, properties=['name'])
            for temp_vm_object in objects:
                if temp_vm_object.obj.name == self.name:
                    vms.append(temp_vm_object.obj)
        elif self.moid:
            vm_obj = VmomiSupport.templateOf('VirtualMachine')(self.module.params['moid'], self.si._stub)
            if vm_obj:
                vms.append(vm_obj)
        if vms:
            if self.params.get('name_match') == 'first':
                self.vm = vms[0]
            elif self.params.get('name_match') == 'last':
                self.vm = vms[-1]
        else:
            self.module.fail_json(msg='Failed to find virtual machine using %s' % (self.name or self.uuid))

    @staticmethod
    def humanize_boot_order(boot_order):
        results = []
        for device in boot_order:
            if isinstance(device, vim.vm.BootOptions.BootableCdromDevice):
                results.append('cdrom')
            elif isinstance(device, vim.vm.BootOptions.BootableDiskDevice):
                results.append('disk')
            elif isinstance(device, vim.vm.BootOptions.BootableEthernetDevice):
                results.append('ethernet')
            elif isinstance(device, vim.vm.BootOptions.BootableFloppyDevice):
                results.append('floppy')
        return results

    def ensure(self):
        boot_order_list = []
        change_needed = False
        kwargs = dict()
        previous_boot_disk = None
        valid_device_strings = ['cdrom', 'disk', 'ethernet', 'floppy']
        self._get_vm()
        for device_order in self.params.get('boot_order'):
            if device_order not in valid_device_strings:
                self.module.fail_json(msg="Invalid device found [%s], please specify device from ['%s']" % (device_order, "', '".join(valid_device_strings)))
            if device_order == 'cdrom':
                first_cdrom = [device for device in self.vm.config.hardware.device if isinstance(device, vim.vm.device.VirtualCdrom)]
                if first_cdrom:
                    boot_order_list.append(vim.vm.BootOptions.BootableCdromDevice())
            elif device_order == 'disk':
                if not self.params.get('boot_hdd_name'):
                    first_hdd = [device for device in self.vm.config.hardware.device if isinstance(device, vim.vm.device.VirtualDisk)]
                else:
                    first_hdd = [device for device in self.vm.config.hardware.device if isinstance(device, vim.vm.device.VirtualDisk) and device.deviceInfo.label == self.params.get('boot_hdd_name')]
                    if not first_hdd:
                        self.module.fail_json(msg="Not found virtual disk with disk label '%s'" % self.params.get('boot_hdd_name'))
                if first_hdd:
                    boot_order_list.append(vim.vm.BootOptions.BootableDiskDevice(deviceKey=first_hdd[0].key))
            elif device_order == 'ethernet':
                first_ether = [device for device in self.vm.config.hardware.device if isinstance(device, vim.vm.device.VirtualEthernetCard)]
                if first_ether:
                    boot_order_list.append(vim.vm.BootOptions.BootableEthernetDevice(deviceKey=first_ether[0].key))
            elif device_order == 'floppy':
                first_floppy = [device for device in self.vm.config.hardware.device if isinstance(device, vim.vm.device.VirtualFloppy)]
                if first_floppy:
                    boot_order_list.append(vim.vm.BootOptions.BootableFloppyDevice())
        if self.params.get('boot_hdd_name'):
            for i in range(0, len(self.vm.config.bootOptions.bootOrder)):
                if isinstance(self.vm.config.bootOptions.bootOrder[i], vim.vm.BootOptions.BootableDiskDevice):
                    if self.vm.config.bootOptions.bootOrder[i].deviceKey:
                        for dev in self.vm.config.hardware.device:
                            if isinstance(dev, vim.vm.device.VirtualDisk) and dev.key == self.vm.config.bootOptions.bootOrder[i].deviceKey:
                                previous_boot_disk = dev.deviceInfo.label
        if len(boot_order_list) != len(self.vm.config.bootOptions.bootOrder):
            kwargs.update({'bootOrder': boot_order_list})
            change_needed = True
        else:
            for i in range(0, len(boot_order_list)):
                boot_device_type = type(boot_order_list[i])
                vm_boot_device_type = type(self.vm.config.bootOptions.bootOrder[i])
                if boot_device_type != vm_boot_device_type:
                    kwargs.update({'bootOrder': boot_order_list})
                    change_needed = True
                elif vm_boot_device_type is vim.vm.BootOptions.BootableDiskDevice and boot_order_list[i].deviceKey != self.vm.config.bootOptions.bootOrder[i].deviceKey:
                    kwargs.update({'bootOrder': boot_order_list})
                    change_needed = True
        if self.params.get('boot_delay') is not None and self.vm.config.bootOptions.bootDelay != self.params.get('boot_delay'):
            kwargs.update({'bootDelay': self.params.get('boot_delay')})
            change_needed = True
        if self.params.get('enter_bios_setup') is not None and self.vm.config.bootOptions.enterBIOSSetup != self.params.get('enter_bios_setup'):
            kwargs.update({'enterBIOSSetup': self.params.get('enter_bios_setup')})
            change_needed = True
        if self.params.get('boot_retry_enabled') is not None and self.vm.config.bootOptions.bootRetryEnabled != self.params.get('boot_retry_enabled'):
            kwargs.update({'bootRetryEnabled': self.params.get('boot_retry_enabled')})
            change_needed = True
        if self.params.get('boot_retry_delay') is not None and self.vm.config.bootOptions.bootRetryDelay != self.params.get('boot_retry_delay'):
            if not self.vm.config.bootOptions.bootRetryEnabled:
                kwargs.update({'bootRetryEnabled': True})
            kwargs.update({'bootRetryDelay': self.params.get('boot_retry_delay')})
            change_needed = True
        boot_firmware_required = False
        if self.params.get('boot_firmware') is not None and self.vm.config.firmware != self.params.get('boot_firmware'):
            change_needed = True
            boot_firmware_required = True
        if self.params.get('secure_boot_enabled') is not None:
            if self.params.get('secure_boot_enabled') and self.params.get('boot_firmware') == 'bios':
                self.module.fail_json(msg='Secure boot cannot be enabled when boot_firmware = bios')
            elif self.params.get('secure_boot_enabled') and self.params.get('boot_firmware') != 'efi' and (self.vm.config.firmware == 'bios'):
                self.module.fail_json(msg="Secure boot cannot be enabled since the VM's boot firmware is currently set to bios")
            elif self.vm.config.bootOptions.efiSecureBootEnabled != self.params.get('secure_boot_enabled'):
                kwargs.update({'efiSecureBootEnabled': self.params.get('secure_boot_enabled')})
                change_needed = True
        changed = False
        results = dict(previous_boot_order=self.humanize_boot_order(self.vm.config.bootOptions.bootOrder), previous_boot_delay=self.vm.config.bootOptions.bootDelay, previous_enter_bios_setup=self.vm.config.bootOptions.enterBIOSSetup, previous_boot_retry_enabled=self.vm.config.bootOptions.bootRetryEnabled, previous_boot_retry_delay=self.vm.config.bootOptions.bootRetryDelay, previous_boot_firmware=self.vm.config.firmware, previous_secure_boot_enabled=self.vm.config.bootOptions.efiSecureBootEnabled, current_boot_order=[])
        if previous_boot_disk:
            results.update({'previous_boot_disk': previous_boot_disk})
        if change_needed:
            vm_conf = vim.vm.ConfigSpec()
            vm_conf.bootOptions = vim.vm.BootOptions(**kwargs)
            if boot_firmware_required:
                vm_conf.firmware = self.params.get('boot_firmware')
            task = self.vm.ReconfigVM_Task(vm_conf)
            try:
                changed, result = wait_for_task(task)
            except TaskError as e:
                self.module.fail_json(msg='Failed to perform reconfigure virtual machine %s for boot order due to: %s' % (self.name or self.uuid, to_native(e)))
        results.update({'current_boot_order': self.humanize_boot_order(self.vm.config.bootOptions.bootOrder), 'current_boot_delay': self.vm.config.bootOptions.bootDelay, 'current_enter_bios_setup': self.vm.config.bootOptions.enterBIOSSetup, 'current_boot_retry_enabled': self.vm.config.bootOptions.bootRetryEnabled, 'current_boot_retry_delay': self.vm.config.bootOptions.bootRetryDelay, 'current_boot_firmware': self.vm.config.firmware, 'current_secure_boot_enabled': self.vm.config.bootOptions.efiSecureBootEnabled})
        if self.params.get('boot_hdd_name'):
            results.update({'current_boot_disk': self.params.get('boot_hdd_name')})
        self.module.exit_json(changed=changed, vm_boot_status=results)