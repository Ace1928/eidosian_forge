from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task, TaskError
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
class PyVmomiHelper(PyVmomi):

    def __init__(self, module):
        super(PyVmomiHelper, self).__init__(module)
        self.device_helper = PyVmomiDeviceHelper(self.module)
        self.sleep_time = 10
        self.controller_types = self.device_helper.scsi_device_type.copy()
        self.controller_types.update(self.device_helper.usb_device_type)
        self.controller_types.update({'sata': self.device_helper.sata_device_type, 'nvme': self.device_helper.nvme_device_type})
        self.config_spec = vim.vm.ConfigSpec()
        self.config_spec.deviceChange = []
        self.change_detected = False
        self.disk_ctl_bus_num_list = dict(sata=list(range(0, 4)), nvme=list(range(0, 4)), scsi=list(range(0, 4)))

    def get_unused_ctl_bus_number(self):
        """
        Get gid of occupied bus numbers of each type of disk controller, update the available bus number list
        """
        for device in self.current_vm_obj.config.hardware.device:
            if isinstance(device, self.device_helper.sata_device_type):
                if len(self.disk_ctl_bus_num_list['sata']) != 0:
                    self.disk_ctl_bus_num_list['sata'].remove(device.busNumber)
            if isinstance(device, self.device_helper.nvme_device_type):
                if len(self.disk_ctl_bus_num_list['nvme']) != 0:
                    self.disk_ctl_bus_num_list['nvme'].remove(device.busNumber)
            if isinstance(device, tuple(self.device_helper.scsi_device_type.values())):
                if len(self.disk_ctl_bus_num_list['scsi']) != 0:
                    self.disk_ctl_bus_num_list['scsi'].remove(device.busNumber)

    def check_ctl_disk_exist(self, ctl_type=None, bus_number=None):
        """
        Check if controller of specified type exists and if there is disk attaching to it
        Return: Specified controller device, True or False of attaching disks
        """
        ctl_specified = None
        disks_attached_exist = False
        if ctl_type is None:
            return (ctl_specified, disks_attached_exist)
        for device in self.current_vm_obj.config.hardware.device:
            if isinstance(device, self.controller_types.get(ctl_type)):
                if bus_number is not None and device.busNumber != bus_number:
                    continue
                ctl_specified = device
                if len(device.device) != 0:
                    disks_attached_exist = True
                break
        return (ctl_specified, disks_attached_exist)

    def create_controller(self, ctl_type, bus_sharing, bus_number=0):
        """
        Create new disk or USB controller with specified type
        Args:
            ctl_type: controller type
            bus_number: disk controller bus number
            bus_sharing: noSharing, virtualSharing, physicalSharing

        Return: Virtual device spec for virtual controller
        """
        if ctl_type == 'sata' or ctl_type == 'nvme' or ctl_type in self.device_helper.scsi_device_type.keys():
            disk_ctl = self.device_helper.create_disk_controller(ctl_type, bus_number, bus_sharing)
        elif ctl_type in self.device_helper.usb_device_type.keys():
            disk_ctl = vim.vm.device.VirtualDeviceSpec()
            disk_ctl.operation = vim.vm.device.VirtualDeviceSpec.Operation.add
            disk_ctl.device = self.device_helper.usb_device_type.get(ctl_type)()
            if ctl_type == 'usb2':
                disk_ctl.device.key = 7000
            elif ctl_type == 'usb3':
                disk_ctl.device.key = 14000
            disk_ctl.device.deviceInfo = vim.Description()
            disk_ctl.device.busNumber = bus_number
        return disk_ctl

    def gather_disk_controller_facts(self):
        """
        Gather existing controller facts

        Return: A dictionary of each type controller facts
        """
        disk_ctl_facts = dict(scsi=dict(), sata=dict(), nvme=dict(), usb2=dict(), usb3=dict())
        for device in self.current_vm_obj.config.hardware.device:
            ctl_facts_dict = dict()
            if isinstance(device, tuple(self.controller_types.values())):
                ctl_facts_dict[device.busNumber] = dict(controller_summary=device.deviceInfo.summary, controller_label=device.deviceInfo.label, controller_busnumber=device.busNumber, controller_controllerkey=device.controllerKey, controller_devicekey=device.key, controller_unitnumber=device.unitNumber, controller_disks_devicekey=device.device)
                if hasattr(device, 'sharedBus'):
                    ctl_facts_dict[device.busNumber]['controller_bus_sharing'] = device.sharedBus
                if isinstance(device, tuple(self.device_helper.scsi_device_type.values())):
                    disk_ctl_facts['scsi'].update(ctl_facts_dict)
                if isinstance(device, self.device_helper.nvme_device_type):
                    disk_ctl_facts['nvme'].update(ctl_facts_dict)
                if isinstance(device, self.device_helper.sata_device_type):
                    disk_ctl_facts['sata'].update(ctl_facts_dict)
                if isinstance(device, self.device_helper.usb_device_type.get('usb2')):
                    disk_ctl_facts['usb2'].update(ctl_facts_dict)
                if isinstance(device, self.device_helper.usb_device_type.get('usb3')):
                    disk_ctl_facts['usb3'].update(ctl_facts_dict)
        return disk_ctl_facts

    def sanitize_disk_controller_config(self):
        """
        Check correctness of controller configuration provided by user

        Return: A list of dictionary with checked controller configured
        """
        if not self.params.get('controllers'):
            self.module.exit_json(changed=False, msg="No controller provided for virtual machine '%s' for management." % self.current_vm_obj.name)
        if 10 != self.params.get('sleep_time') <= 300:
            self.sleep_time = self.params.get('sleep_time')
        exec_get_unused_ctl_bus_number = False
        controller_config = self.params.get('controllers')
        for ctl_config in controller_config:
            if ctl_config:
                if ctl_config['type'] not in self.device_helper.usb_device_type.keys():
                    if ctl_config['state'] == 'absent' and ctl_config.get('controller_number') is None:
                        self.module.fail_json(msg='Disk controller number is required when removing it.')
                    if ctl_config['state'] == 'present' and (not exec_get_unused_ctl_bus_number):
                        self.get_unused_ctl_bus_number()
                        exec_get_unused_ctl_bus_number = True
                if ctl_config['state'] == 'present' and ctl_config['type'] == 'nvme':
                    vm_hwv = int(self.current_vm_obj.config.version.split('-')[1])
                    if vm_hwv < 13:
                        self.module.fail_json(msg="Can not create new NVMe disk controller due to VM hardware version is '%s', not >= 13." % vm_hwv)
        if exec_get_unused_ctl_bus_number:
            for ctl_config in controller_config:
                if ctl_config and ctl_config['state'] == 'present' and (ctl_config['type'] not in self.device_helper.usb_device_type.keys()):
                    if ctl_config['type'] in self.device_helper.scsi_device_type.keys():
                        if len(self.disk_ctl_bus_num_list['scsi']) != 0:
                            ctl_config['controller_number'] = self.disk_ctl_bus_num_list['scsi'].pop(0)
                        else:
                            ctl_config['controller_number'] = None
                    elif ctl_config['type'] == 'sata' or ctl_config['type'] == 'nvme':
                        if len(self.disk_ctl_bus_num_list.get(ctl_config['type'])) != 0:
                            ctl_config['controller_number'] = self.disk_ctl_bus_num_list.get(ctl_config['type']).pop(0)
                        else:
                            ctl_config['controller_number'] = None
        return controller_config

    def configure_disk_controllers(self):
        """
        Do disk controller management, add or remove

        Return: Operation result
        """
        if self.params['gather_disk_controller_facts']:
            results = {'changed': False, 'failed': False, 'disk_controller_data': self.gather_disk_controller_facts()}
            return results
        controller_config = self.sanitize_disk_controller_config()
        for disk_ctl_config in controller_config:
            if disk_ctl_config and disk_ctl_config['state'] == 'present':
                if disk_ctl_config['type'] in self.device_helper.usb_device_type.keys():
                    usb_exists, has_disks_attached = self.check_ctl_disk_exist(disk_ctl_config['type'])
                    if usb_exists:
                        self.module.warn("'%s' USB controller already exists, can not add more." % disk_ctl_config['type'])
                    else:
                        disk_controller_new = self.create_controller(disk_ctl_config['type'], disk_ctl_config.get('bus_sharing'))
                        self.config_spec.deviceChange.append(disk_controller_new)
                        self.change_detected = True
                elif disk_ctl_config.get('controller_number') is not None:
                    disk_controller_new = self.create_controller(disk_ctl_config['type'], disk_ctl_config.get('bus_sharing'), disk_ctl_config.get('controller_number'))
                    self.config_spec.deviceChange.append(disk_controller_new)
                    self.change_detected = True
                elif disk_ctl_config['type'] in self.device_helper.scsi_device_type.keys():
                    self.module.warn("Already 4 SCSI controllers, can not add new '%s' controller." % disk_ctl_config['type'])
                else:
                    self.module.warn("Already 4 '%s' controllers, can not add new one." % disk_ctl_config['type'])
            elif disk_ctl_config and disk_ctl_config['state'] == 'absent':
                existing_ctl, has_disks_attached = self.check_ctl_disk_exist(disk_ctl_config['type'], disk_ctl_config.get('controller_number'))
                if existing_ctl is not None:
                    if not has_disks_attached:
                        ctl_spec = vim.vm.device.VirtualDeviceSpec()
                        ctl_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.remove
                        ctl_spec.device = existing_ctl
                        self.config_spec.deviceChange.append(ctl_spec)
                        self.change_detected = True
                    else:
                        self.module.warn("Can not remove specified controller, type '%s', bus number '%s', there are disks attaching to it." % (disk_ctl_config['type'], disk_ctl_config.get('controller_number')))
                else:
                    self.module.warn("Can not find specified controller to remove, type '%s', bus number '%s'." % (disk_ctl_config['type'], disk_ctl_config.get('controller_number')))
        try:
            task = self.current_vm_obj.ReconfigVM_Task(spec=self.config_spec)
            wait_for_task(task)
        except vim.fault.InvalidDeviceSpec as e:
            self.module.fail_json(msg='Failed to configure controller on given virtual machine due to invalid device spec : %s' % to_native(e.msg), details='Please check ESXi server logs for more details.')
        except vim.fault.RestrictedVersion as e:
            self.module.fail_json(msg='Failed to reconfigure virtual machine due to product versioning restrictions: %s' % to_native(e.msg))
        except TaskError as task_e:
            self.module.fail_json(msg=to_native(task_e))
        if task.info.state == 'error':
            results = {'changed': self.change_detected, 'failed': True, 'msg': task.info.error.msg}
        else:
            if self.change_detected:
                time.sleep(self.sleep_time)
            results = {'changed': self.change_detected, 'failed': False, 'disk_controller_data': self.gather_disk_controller_facts()}
        return results