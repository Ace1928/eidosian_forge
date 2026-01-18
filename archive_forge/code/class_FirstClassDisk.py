from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, TaskError, vmware_argument_spec, wait_for_task
from ansible.module_utils._text import to_native
class FirstClassDisk(PyVmomi):

    def __init__(self, module):
        super(FirstClassDisk, self).__init__(module)
        self.datacenter_name = self.params['datacenter_name']
        self.datastore_name = self.params['datastore_name']
        self.disk_name = self.params['disk_name']
        self.desired_state = module.params['state']
        self.size_mb = None
        if self.params['size']:
            size_regex = re.compile('(\\d+)([MGT]B)')
            disk_size_m = size_regex.match(self.params['size'])
            if disk_size_m:
                number = disk_size_m.group(1)
                unit = disk_size_m.group(2)
            else:
                self.module.fail_json(msg='Failed to parse disk size, please review value provided using documentation.')
            number = int(number)
            if unit == 'GB':
                self.size_mb = 1024 * number
            elif unit == 'TB':
                self.size_mb = 1048576 * number
            else:
                self.size_mb = number
        self.datastore_obj = self.find_datastore_by_name(datastore_name=self.datastore_name, datacenter_name=self.datacenter_name)
        if not self.datastore_obj:
            self.module.fail_json(msg='Failed to find datastore %s.' % self.datastore_name)
        self.disk = self.find_first_class_disk_by_name(self.disk_name, self.datastore_obj)

    def create_fcd_result(self, state):
        result = dict(name=self.disk.config.name, datastore_name=self.disk.config.backing.datastore.name, size_mb=self.disk.config.capacityInMB, state=state)
        return result

    def create(self):
        result = dict(changed=False)
        if not self.disk:
            result['changed'] = True
            if not self.module.check_mode:
                backing_spec = vim.vslm.CreateSpec.DiskFileBackingSpec()
                backing_spec.datastore = self.datastore_obj
                vslm_create_spec = vim.vslm.CreateSpec()
                vslm_create_spec.backingSpec = backing_spec
                vslm_create_spec.capacityInMB = self.size_mb
                vslm_create_spec.name = self.disk_name
                try:
                    if self.is_vcenter():
                        task = self.content.vStorageObjectManager.CreateDisk_Task(vslm_create_spec)
                    else:
                        task = self.content.vStorageObjectManager.HostCreateDisk_Task(vslm_create_spec)
                    changed, self.disk = wait_for_task(task)
                except vmodl.RuntimeFault as runtime_fault:
                    self.module.fail_json(msg=to_native(runtime_fault.msg))
                except vmodl.MethodFault as method_fault:
                    self.module.fail_json(msg=to_native(method_fault.msg))
                except TaskError as task_e:
                    self.module.fail_json(msg=to_native(task_e))
                except Exception as generic_exc:
                    self.module.fail_json(msg='Failed to create disk due to generic exception %s' % to_native(generic_exc))
                result['diff'] = {'before': {}, 'after': {}}
                result['diff']['before']['first_class_disk'] = self.create_fcd_result('absent')
                result['diff']['after']['first_class_disk'] = self.create_fcd_result('present')
                result['first_class_disk'] = result['diff']['after']['first_class_disk']
        elif self.size_mb < self.disk.config.capacityInMB:
            self.module.fail_json(msg='Given disk size is smaller than current size (%dMB < %dMB). Reducing disks is not allowed.' % (self.size_mb, self.disk.config.capacityInMB))
        elif self.size_mb > self.disk.config.capacityInMB:
            result['changed'] = True
            if not self.module.check_mode:
                result['diff'] = {'before': {}, 'after': {}}
                result['diff']['before']['first_class_disk'] = self.create_fcd_result('present')
                try:
                    if self.is_vcenter():
                        task = self.content.vStorageObjectManager.ExtendDisk_Task(self.disk.config.id, self.datastore_obj, self.size_mb)
                    else:
                        task = self.content.vStorageObjectManager.HostExtendDisk_Task(self.disk.config.id, self.datastore_obj, self.size_mb)
                    wait_for_task(task)
                except vmodl.RuntimeFault as runtime_fault:
                    self.module.fail_json(msg=to_native(runtime_fault.msg))
                except vmodl.MethodFault as method_fault:
                    self.module.fail_json(msg=to_native(method_fault.msg))
                except TaskError as task_e:
                    self.module.fail_json(msg=to_native(task_e))
                except Exception as generic_exc:
                    self.module.fail_json(msg='Failed to increase disk size due to generic exception %s' % to_native(generic_exc))
                self.disk = self.find_first_class_disk_by_name(self.disk_name, self.datastore_obj)
                result['diff']['after']['first_class_disk'] = self.create_fcd_result('present')
                result['first_class_disk'] = result['diff']['after']['first_class_disk']
        self.module.exit_json(**result)

    def delete(self):
        result = dict(changed=False)
        if self.disk:
            result['changed'] = True
            if not self.module.check_mode:
                result['diff'] = {'before': {}, 'after': {}}
                result['diff']['before']['first_class_disk'] = self.create_fcd_result('present')
                result['diff']['after']['first_class_disk'] = self.create_fcd_result('absent')
                result['first_class_disk'] = result['diff']['after']['first_class_disk']
                try:
                    if self.is_vcenter():
                        task = self.content.vStorageObjectManager.DeleteVStorageObject_Task(self.disk.config.id, self.datastore_obj)
                    else:
                        task = self.content.vStorageObjectManager.HostDeleteVStorageObject_Task(self.disk.config.id, self.datastore_obj)
                    wait_for_task(task)
                except vmodl.RuntimeFault as runtime_fault:
                    self.module.fail_json(msg=to_native(runtime_fault.msg))
                except vmodl.MethodFault as method_fault:
                    self.module.fail_json(msg=to_native(method_fault.msg))
                except TaskError as task_e:
                    self.module.fail_json(msg=to_native(task_e))
                except Exception as generic_exc:
                    self.module.fail_json(msg='Failed to delete disk due to generic exception %s' % to_native(generic_exc))
        self.module.exit_json(**result)